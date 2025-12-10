import cv2

# 0, 1, 2, 3 යන අංක උත්සාහ කර බලමු
for index in range(4):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Checking camera at index {index}...")
        ret, frame = cap.read()
        if ret:
            print(f"SUCCESS: Camera found at index {index}!")
            cv2.imshow(f'Test Cam Index {index}', frame)
            cv2.waitKey(2000) # තත්පර 2ක් පෙන්වයි
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"No camera at index {index}")

print("Test finished. Press any key to exit.")