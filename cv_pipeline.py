import numpy as np
import cv2

class GrabCutPipeline:
    def __init__(self: object,
                 image: np.ndarray) -> None:
        self.image = image
        self.output, self.mask = self.initialize_image(image)
        self.drawing = False  # True if the mouse is pressed
        self.mode = True  # True for foreground, False for background
        self.ix, self.iy = -1, -1
        self.drawings = []  # List to save drawings
        self.initial_boundary = 0  # Tuple to save the initial boundary

    def initialize_image(self: object,
                         image: np.ndarray) -> tuple:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        output = image.copy()
        # Binarization helps a bit with the initial removal of the background
        binarized_image = cv2.adaptiveThreshold(
            gray_image,
            maxValue=1,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=9,
            C=7
        )
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[:] = cv2.GC_PR_BGD
        mask[binarized_image == 0] = cv2.GC_FGD
        return output, mask

    def draw(self: object,
             event,
             x,
             y,
             flags,
             param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_circle(x, y)
                self.save_drawing(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.draw_circle(x, y)
            self.save_drawing(x, y)

    def draw_circle(self: object,
                    x,
                    y) -> None:
        color = (255, 255, 255) if self.mode else (0, 0, 0)
        gc_flag = cv2.GC_FGD if self.mode else cv2.GC_BGD
        cv2.circle(self.output, (x, y), 3, color, -1)
        cv2.circle(self.mask, (x, y), 3, gc_flag, -1)

    def save_drawing(self: object,
                     x,
                     y) -> None:
        self.drawings.append((x, y, self.mode))

    def perform_grabcut(self: object,
                        roi: tuple) -> None:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, self.mask, roi, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((self.mask == cv2.GC_PR_BGD) | (self.mask == cv2.GC_BGD), 0, 1).astype('uint8')
        self.initial_output = self.image * mask2[:, :, np.newaxis]

    def save_initial_boundary(self: object,
                              roi: tuple) -> None:
        self.initial_boundary = roi

    def run_grabcut(self: object) -> None:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, self.mask, self.initial_boundary, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == cv2.GC_PR_BGD) | (self.mask == cv2.GC_BGD), 0, 1).astype('uint8')
        self.output = self.image * mask2[:, :, np.newaxis]

    def plot_drawings_on_image(self:  object) -> None:
        # Draw the initial boundary
        for i in range(len(self.initial_boundary) - 1):
            cv2.line(self.output, self.initial_boundary[i], self.initial_boundary[i + 1], (0, 255, 0), 2)

        # Draw the additional user drawings
        for x, y, mode in self.drawings:
            color = (255, 255, 255) if mode else (0, 0, 0)
            cv2.circle(self.output, (x, y), 3, color, -1)

    def process_image(self: object) -> None:
        # Select ROI for initial grabcut
        roi = cv2.selectROI('image', self.output)
        cv2.destroyAllWindows()
        self.save_initial_boundary(roi)
        self.perform_grabcut(roi)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw)

        # Main loop
        while True:
            cv2.imshow('image', self.output)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC to exit
                break
            elif k == ord('0'):  # Press '0' to mark background
                self.mode = False
            elif k == ord('1'):  # Press '1' to mark foreground
                self.mode = True
            elif k == ord('r'):  # Press 'r' to run grabcut again
                self.run_grabcut()

        cv2.destroyAllWindows()