rm compare.mkv
ffmpeg -i output.mkv -i output_flow.mkv -filter_complex hstack -c:v libx264 -crf 18 up.mkv
ffmpeg -i output_depth.mkv -i output_obs.mkv -filter_complex hstack -c:v libx264 -crf 18 down.mkv
ffmpeg -i up.mkv -i down.mkv -filter_complex vstack -c:v libx264 -crf 18 compare.mkv
rm up.mkv down.mkv
