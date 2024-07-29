# Computer vision algorithms for background removal

## How to use app.py
1. Launch it
2. Select the model you want to use
   - GrabCut
   - BiRefNet
3. If you selected BiRefNet, just upload the picture and wait for the results.
4. If you selected GrabCut, after uploading the picture, a pop-up will appear.
   - Draw a rectangle around the object you want to keep.
   - Press 'ENTER'
   - A new pop-up will appear.
   - Press 'r' to apply the initial GrabCut.
   - If you need to update the mask:
     - Draw with the mouse on the image.
     - Press '1' for the foreground.
     - Press '0' for the background.
     - Press 'r' to apply your drawings.
   - To exit press 'esc'.