import pandas as pd

# Paths for the two datasets
csv1 = "/run/media/soumi/Devansh/test_llava3_output.csv"           # Dataset 1
csv2 = "/run/media/soumi/Devansh/test_llava_from_csv_output.csv"   # Dataset 2
output_csv = "/run/media/soumi/Devansh/combined_windows.csv"

# Load both datasets
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# Ensure timestamps are numeric (remove trailing 's')
df1['Sec'] = df1['Timestamp'].str.replace('s', '', regex=False).astype(int)
df2['Sec'] = df2['Timestamp'].str.replace('s', '', regex=False).astype(int)

combined_rows = []

# Process each video
for video in df1['VideoName'].unique():
    vid1 = df1[df1['VideoName'] == video].sort_values('Sec')
    vid2 = df2[df2['VideoName'] == video].sort_values('Sec')

    if vid1.empty or vid2.empty:
        continue

    max_time = min(vid1['Sec'].max(), vid2['Sec'].max())

    # Sliding window: every second, take 4s from Dataset1 and next sec from Dataset2
    for start in range(0, max_time - 4):
        end = start + 3
        next_sec = end + 1

        # Captions from Dataset1: 4s window
        window1 = vid1[(vid1['Sec'] >= start) & (vid1['Sec'] <= end)]
        # Caption from Dataset2: next second
        window2 = vid2[vid2['Sec'] == next_sec]

        if not window1.empty and not window2.empty:
            # Format dataset1 captions with image names
            ds1_captions = "\n".join(
                [f"{row.Sec}s: {row.Caption}  [Image: {row.ImageFile}]"
                 for row in window1.itertuples()]
            )
            # Dataset2 caption with image name
            ds2_caption = f"{next_sec}s: {window2['Caption'].iloc[0]}  [Image: {window2['ImageFile'].iloc[0]}]"

            combined_rows.append({
                "VideoName": video,
                "Window": f"{start}s-{end}s",
                "Dataset1_Captions": ds1_captions,
                "Dataset2_Caption": ds2_caption
            })

# Save combined dataset
df_out = pd.DataFrame(combined_rows)
df_out.to_csv(output_csv, index=False)

print(f"? Combined sliding-window dataset with image names saved to: {output_csv}")
