import cv2

video_path= "C:/Users/RukaSuirenji/Downloads/colmap-x64-windows-nocuda/v2/20251003_141708.mp4"
output_path = "C:/Users/RukaSuirenji/Downloads/colmap-x64-windows-nocuda1/test/"

cap = cv2.VideoCapture(video_path)

#畫面設定
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

id = 0
write = 1
while cap.isOpened():
    ret, frame = cap.read()
    if (write % 16 == 0) and ret:
        cv2.imwrite(output_path + str(id) + '_frame.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        write += 1
        id += 1
    else:
        write += 1
