import cv2
import onnxruntime
import numpy as np


class MPCount():
    def __init__(self, modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(modelpath, so)
        print('input_shape:', self.net.get_inputs()[0].shape)    ####输入高宽是动态不固定的
        self.unit_size=16
        self.log_para=1000
        self.input_name = self.net.get_inputs()[0].name
    
    def preprocess(self, frame):
        h, w = frame.shape[:2]
        new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
        new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h
        if h >= new_h:
            top = 0
            bottom = 0
        else:
            dh = new_h - h
            top = dh // 2
            bottom = dh // 2 + dh % 2
            
        if w >= new_w:
            left = 0
            right = 0
        else:
            dw = new_w - w
            left = dw // 2
            right = dw // 2 + dw % 2

        padded_image = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) 
        x = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        return x, (left, top, right, bottom)
        
    def detect(self, frame):
        x, (left, top, right, bottom) = self.preprocess(frame)

        x = ((x.astype(np.float32) / 255) - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[np.newaxis, ...]

        result, _ = self.net.run(None, {self.input_name: x})

        result_map = result[:, :, top:result.shape[2] - bottom, left:result.shape[3] - right]
        return result_map, int(np.sum(result_map) / self.log_para)
    
def draw_result(image, result_map, people_count):
    image_width, image_height = image.shape[1], image.shape[0]
    drawimg = result_map[0, 0].copy()

    # Apply ColorMap
    drawimg = (drawimg - drawimg.min()) / (drawimg.max() - drawimg.min() + 1e-5)
    drawimg = (drawimg * 255).astype(np.uint8)
    drawimg = cv2.applyColorMap(drawimg, cv2.COLORMAP_JET)

    drawimg = cv2.resize(drawimg, dsize=(image_width, image_height))

    # addWeighted
    drawimg = cv2.addWeighted(image, 0.35, drawimg, 0.65, 1.0)

    # Peaple Count
    cv2.putText(drawimg, "People Count : " + str(people_count), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return drawimg

if __name__=='__main__':
    mynet = MPCount('weights/MPCount_qnrf.onnx')
    imgpath = "0.jpg"
    
    frame = cv2.imread(imgpath)
    result_map, people_count = mynet.detect(frame)
    drawimg = draw_result(frame, result_map, people_count)

    cv2.imwrite("result.jpg", drawimg)
    # cv2.namedWindow('MPCount Demo : Original Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('MPCount Demo : Original Image', frame)
    # cv2.namedWindow('MPCount Demo : Activation Map', cv2.WINDOW_NORMAL)
    # cv2.imshow('MPCount Demo : Activation Map', drawimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
