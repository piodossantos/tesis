FILES="data/*"
mkdir data_numbers
mkdir data_numbers/data
for f in $FILES
do
	echo "Processing $f"
    ffmpeg -i $f -vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy data_numbers/$f
done