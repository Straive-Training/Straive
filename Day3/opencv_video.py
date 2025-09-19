import cv2

video_path = 'face.mp4'

video = cv2.VideoCapture(video_path)
cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    check, frame = video.read()

    if not check:
        print("End of video or can't read the frame.")
        break

    grey_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.2, minNeighbors=5)
    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('Video Playback', frame)

    # Wait 25 ms, adjust if video plays too fast or too slowf
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

