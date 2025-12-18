import argparse
import os

import cv2
import numpy as np

import platform

IS_WINDOWS = platform.system() == 'Windows'


def load_csv(path, delimiter=',', offset=0, dtype=np.float64):
    with open(path, 'r') as f:
        lines = map(lambda _: _.strip(), f.readlines()[offset:])
        lines = [list(map(float, _.replace(' ', '').split(delimiter)[1:])) for _ in lines]
    return np.asarray(lines, dtype=dtype)


def resize_to_ratio(img, desired_ratio=1.5, max_width=None):
    h, w = img.shape[:2]
    ratio = w / h
    if ratio > desired_ratio:
        img = cv2.resize(img, dsize=(int(h * 1.5), h), interpolation=cv2.INTER_AREA)
    elif ratio < desired_ratio:
        img = cv2.resize(img, dsize=(w, int(w / 1.5)), interpolation=cv2.INTER_AREA)
    if max_width:
        img = cv2.resize(img, dsize=(max_width, int(max_width / desired_ratio)), interpolation=cv2.INTER_AREA)
    return img


def x1y1x2y2toxywh(x1, y1, x2, y2):
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


def put_text(img, x, y, txt, color=(0, 0, 0), txt_color=(255, 255, 255), margin=2, width=None):
    if width is None:
        (w_m, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
    else:
        w_m = width
    x_min, x_max = x, x + w_m + 2 * margin
    y_min, y_max = y, y + 20
    sub_img = img[y_min:y_max, x_min:x_max]
    box = np.full(sub_img.shape, fill_value=color, dtype=sub_img.dtype)
    img[y_min:y_max, x_min:x_max] = cv2.addWeighted(sub_img, 0.5, box, 0.5, 1.0)
    # cv2.rectangle(img, (x, y), (x + w_m, y + 20), color, -1)
    cv2.putText(img, txt, (x + margin, y + 15), cv2.FONT_HERSHEY_COMPLEX,
                0.5, txt_color, 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('scene', type=str)
    parser.add_argument('img_id', type=int)
    parser.add_argument('--output_path', type=str, default='video')
    parser.add_argument('--fps', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_path

    model_path = f'output/multilevel_interp/{args.dataset}/{args.scene}/L=8_min=50000_max=0.85_compress=True'
    result_path = os.path.join(model_path, 'results.csv')
    result = load_csv(result_path)

    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = None

    # x1, y1, x2, y2 = 575, 110, 995, 500
    render_path = os.path.join(model_path, 'renders')
    anchors = np.linspace(0, 1, 50)[::7]
    for i, r in enumerate(np.linspace(0, 1, 50)):
        img_file = os.path.join(render_path, f'r={r:.03f}', f'{args.img_id}.png')
        img = cv2.imread(img_file)
        img = resize_to_ratio(img, 1.5, max_width=600)
        h, w = img.shape[:2]
        if out is None:
            out = cv2.VideoWriter(os.path.join(output_path, f'{args.scene}.mp4'), fourcc, args.fps, (w, h))

        psnr = result[i, 0]
        ssim = result[i, 1]
        lpips = result[i, 2]
        fps = result[i, 4]
        size = result[i, -1]

        print(f'r={r:.03f}, psnr={psnr:.03f}, ssim={ssim:.03f}, lpips={lpips:.03f}, fps={fps:.02f}')

        alpha_txt = f'alpha={r:.03f}'
        psnr_txt = f'PSNR: {psnr:.03f}'
        size_txt = f'Size: {size:.03f} (MB)'
        fps_txt = f'FPS: {fps:.02f}'
        txt_color = (255, 255, 255)
        color = (0, 0, 0)
        psnr_color = (97, 187, 157)
        size_color = (184, 177, 134)
        fps_color = (101, 217, 255)
        x, y = 10, 10
        box_width = max(cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0][0]
                        for txt in [psnr_txt, size_txt, fps_txt])
        put_text(img, x, y, alpha_txt, color=color,
                 txt_color=(255, 255, 255),  # if not np.isin(r, anchors) else (0, 0, 255),
                 width=box_width)
        put_text(img, x, y + 20, psnr_txt, color=color, txt_color=psnr_color, width=box_width)
        put_text(img, x, y + 40, size_txt, color=color, txt_color=size_color, width=box_width)
        put_text(img, x, y + 60, fps_txt, color=color, txt_color=fps_color, width=box_width)

        if IS_WINDOWS:
            cv2.imshow('img', img)
            cv2.waitKey(20)
        if out is not None:
            out.write(img)
    if IS_WINDOWS:
        cv2.destroyAllWindows()
    if out is not None:
        out.release()
        print(f'saved video for {args.scene}/{args.img_id}')


if __name__ == '__main__':
    main()
