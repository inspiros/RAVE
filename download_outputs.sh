rsync -rav -e 'C:/msys64/usr/bin/ssh.exe -A -J nhatth@attilio.enst.fr' \
  --include='*/' --include '*.csv' --exclude '*' \
  nhatth@10.42.0.241:/home/nhatth/code/projects/GoDe/Scaffold-GS/output .

#rsync -rav -e 'C:/msys64/usr/bin/ssh.exe -A -J nhatth@attilio.enst.fr' \
#  --include='*/' --include '*.png' --exclude '*' \
#  nhatth@10.42.0.241:/home/nhatth/code/projects/GoDe/Scaffold-GS/output/multilevel_interp/mipnerf360/stump output/multilevel_interp/mipnerf360
