# Concatenates all mp4 files in a given directory.
cd $1
ls *.mp4 | xargs -I {} echo file \'{}\' > concat_spec.txt
ffmpeg -f concat -i concat_spec.txt -c copy concat.mp4

