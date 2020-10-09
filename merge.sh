rm compare.mkv
ffmpeg -i output.mkv -i output_flow.mkv -filter_complex hstack -c:v libx264 -crf 16 up.mkv
ffmpeg -i output_noRotateFlow.mkv -i output_depth.mkv -filter_complex hstack -c:v libx264 -crf 16 down.mkv
ffmpeg -i up.mkv -i down.mkv -filter_complex vstack -c:v libx264 -crf 16 compare.mkv
rm up.mkv down.mkv
