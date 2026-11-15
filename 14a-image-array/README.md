# 14A｜影像就是陣列：從照片到模型輸入

一張照片在程式裡是 (高, 寬, 3) 的 uint8 陣列：像素、裁切、亮度、uint8 溢位、OpenCV 的 BGR、縮放，最後做出模型要的 (1, 3, 224, 224)。

    python 14a-image-array/image_array.py
