import argparse
import os
import cv2
import numpy as np


class BaseRemover:
    def __init__(self, rgb_img, seg_img, output_dir):
        self.rgb_img = rgb_img
        self.seg_img = seg_img
        self.output_dir = output_dir
        self.result = None
        self.bbox = None

    def _get_mask(self, coord):
        """通用掩膜生成方法"""
        x, y = coord
        target_label = self.seg_img[y, x]

        base_mask = np.where(self.seg_img == target_label, 255, 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(base_mask)
        target_label_id = labels[y, x]
        return np.where(labels == target_label_id, 255, 0).astype(np.uint8)

    def _calculate_bbox(self, mask, pad=20):
        """通用边界框计算"""
        mask_indices = np.where(mask == 255)
        if len(mask_indices[0]) == 0:
            return None

        y_min = np.min(mask_indices[0])
        y_max = np.max(mask_indices[0])
        x_min = np.min(mask_indices[1])
        x_max = np.max(mask_indices[1])

        h, w = self.seg_img.shape[:2]
        return (
            max(0, x_min - pad),
            max(0, y_min - pad),
            min(w - 1, x_max + pad),
            min(h - 1, y_max + pad)
        )


class ObjectRemover(BaseRemover):
    """原始物体移除功能"""

    def process(self, coord):
        precise_mask = self._get_mask(coord)
        white_bg = np.full_like(self.rgb_img, 255, dtype=np.uint8)

        # 生成处理结果
        self.result = cv2.bitwise_and(self.rgb_img, self.rgb_img, mask=~precise_mask)
        self.result += cv2.bitwise_and(white_bg, white_bg, mask=precise_mask)

        # 计算边界框
        self.bbox = self._calculate_bbox(precise_mask)
        return self.result


class ObjectMover(BaseRemover):
    """物体移动"""

    def __init__(self, rgb_img, seg_img, output_dir):
        super().__init__(rgb_img, seg_img, output_dir)
        self.source_coord = None
        self.object_img = None
        self.object_mask = None
        self.object_center = None  # 添加物体中心点存储

    def process(self, coord):
        if self.source_coord is None:
            return self._capture_object(coord)
        else:
            return self._move_object(coord)

    def _capture_object(self, coord):
        """第一阶段：捕获要移动的物体"""
        self.source_coord = coord
        precise_mask = self._get_mask(coord)

        # 保存物体图像和掩膜
        self.object_mask = precise_mask
        self.object_img = cv2.bitwise_and(self.rgb_img, self.rgb_img, mask=precise_mask)

        # 计算物体中心
        mask_indices = np.where(precise_mask == 255)
        if len(mask_indices[0]) > 0:
            y_center = int(np.mean(mask_indices[0]))
            x_center = int(np.mean(mask_indices[1]))
            self.object_center = (x_center, y_center)
        else:
            self.object_center = coord

        # 显示预览
        preview = self.rgb_img.copy()
        cv2.circle(preview, coord, 12, (0, 255, 0, 255), -1)
        return preview

    def _move_object(self, target_coord):
        """第二阶段：执行移动操作"""
        # 创建目标图像 - 首先擦除原始位置
        white_bg = np.full_like(self.rgb_img, 255, dtype=np.uint8)
        target_img = cv2.bitwise_and(self.rgb_img, self.rgb_img, mask=~self.object_mask)
        target_img += cv2.bitwise_and(white_bg, white_bg, mask=self.object_mask)

        # 计算物体偏移量
        if self.object_center is None:
            self.object_center = self.source_coord

        # 计算从物体中心到目标位置的偏移量
        dx = target_coord[0] - self.object_center[0]
        dy = target_coord[1] - self.object_center[1]

        # 获取物体尺寸和位置
        mask_indices = np.where(self.object_mask == 255)
        if len(mask_indices[0]) == 0:
            self.result = target_img
            return target_img

        y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
        x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])

        obj_height = y_max - y_min + 1
        obj_width = x_max - x_min + 1

        # 计算新位置
        new_x_min = x_min + dx
        new_y_min = y_min + dy

        # 边界检查
        h, w = target_img.shape[:2]
        if new_x_min < 0:
            new_x_min = 0
        if new_y_min < 0:
            new_y_min = 0
        if new_x_min + obj_width > w:
            new_x_min = w - obj_width
        if new_y_min + obj_height > h:
            new_y_min = h - obj_height

        # 提取物体区域
        obj_roi = self.object_img[y_min:y_max + 1, x_min:x_max + 1]
        mask_roi = self.object_mask[y_min:y_max + 1, x_min:x_max + 1]

        # 应用到新位置
        new_x_max = new_x_min + obj_width
        new_y_max = new_y_min + obj_height

        # 确保新位置在图像范围内
        if (new_x_min >= 0 and new_y_min >= 0 and
                new_x_max < w and new_y_max < h):
            # 创建新位置的掩膜
            new_mask = np.zeros_like(self.object_mask)
            new_mask[new_y_min:new_y_max, new_x_min:new_x_max] = mask_roi

            # 应用掩膜到新位置
            target_area = target_img[new_y_min:new_y_max, new_x_min:new_x_max]
            target_bg = cv2.bitwise_and(target_area, target_area, mask=~mask_roi)
            target_fg = cv2.bitwise_and(obj_roi, obj_roi, mask=mask_roi)
            target_img[new_y_min:new_y_max, new_x_min:new_x_max] = target_bg + target_fg

            # 更新边界框计算
            self.bbox = (
                max(0, new_x_min - 20),
                max(0, new_y_min - 20),
                min(w - 1, new_x_max + 20),
                min(h - 1, new_y_max + 20)
            )

        self.result = target_img
        return self.result


class DynamicObjectProcessor:
    def __init__(self, rgb_path, seg_path, output_dir, function_type):
        self.rgb_path = rgb_path
        self.seg_path = seg_path
        self.output_dir = output_dir
        self.function_type = function_type

        # 初始化图像
        self._load_images()
        os.makedirs(self.output_dir, exist_ok=True)
        self._save_original()

        # 初始化功能模块
        self.processor = self._init_processor()
        self.target_coord = None

    def _load_images(self):
        bgr_img = cv2.imread(self.rgb_path)
        if bgr_img is None:
            raise ValueError("无法读取原始图像")
        self.rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2BGRA)

        self.seg_img = cv2.imread(self.seg_path, cv2.IMREAD_GRAYSCALE)
        if self.seg_img is None:
            raise ValueError("无法读取分割图像")

        if self.rgb_img.shape[:2] != self.seg_img.shape:
            raise ValueError("图像尺寸不匹配")

    def _save_original(self):
        original_name = f"32bit_original_{os.path.basename(self.rgb_path)}"
        output_path = os.path.join(self.output_dir, original_name)
        cv2.imwrite(output_path, self.rgb_img)
        print(f"原始图像已保存：{output_path}")

    def _init_processor(self):
        """根据配置初始化功能模块"""
        if self.function_type == "remove":
            return ObjectRemover(self.rgb_img, self.seg_img, self.output_dir)
        elif self.function_type == "move":
            return ObjectMover(self.rgb_img, self.seg_img, self.output_dir)
        else:
            raise ValueError("不支持的函数类型")

    def _show_results(self, result_img):
        """显示处理结果"""
        display_img = cv2.cvtColor(result_img, cv2.COLOR_BGRA2BGR)
        cv2.imshow("处理效果", display_img)
        cv2.waitKey(500)

    def _save_cropped_images(self):
        """保存局部裁剪图像"""
        if self.processor.bbox is None:
            return

        x1, y1, x2, y2 = self.processor.bbox
        size = min(x2 - x1, y2 - y1)
        x2 = x1 + size
        y2 = y1 + size

        # 边界检查
        h, w = self.rgb_img.shape[:2]
        x2 = min(x2, w - 1)
        y2 = min(y2, h - 1)

        # 保存裁剪图像
        cropped_original = self.rgb_img[y1:y2, x1:x2]
        cropped_result = self.processor.result[y1:y2, x1:x2]

        original_path = os.path.join(self.output_dir, f"cropped_original_{os.path.basename(self.rgb_path)}")
        result_path = os.path.join(self.output_dir, f"cropped_result_{os.path.basename(self.rgb_path)}")
        cv2.imwrite(original_path, cropped_original)
        cv2.imwrite(result_path, cropped_result)
        print(f"局部图像已保存：{original_path}, {result_path}")

    def _save_result(self):
        """保存最终结果"""
        result_name = f"32bit_{self.function_type}_result_{os.path.basename(self.rgb_path)}"
        output_path = os.path.join(self.output_dir, result_name)
        cv2.imwrite(output_path, self.processor.result)
        print(f"处理结果已保存：{output_path}")
        self._save_cropped_images()

    def run(self):
        cv2.namedWindow("Interaction Window")
        cv2.setMouseCallback("Interaction Window", self._mouse_handler)

        while True:
            # 显示当前图像状态
            if self.processor.result is not None:
                display_img = cv2.cvtColor(self.processor.result, cv2.COLOR_BGRA2BGR)
            else:
                display_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGRA2BGR)

            cv2.imshow("Interaction Window", display_img)

            key = cv2.waitKey(20)
            if key == 27:  # ESC退出
                break
            elif key == ord('r'):  # 重置
                self.target_coord = None
                if isinstance(self.processor, ObjectMover):
                    self.processor.source_coord = None
                self.processor.result = None

        cv2.destroyAllWindows()
        if self.processor.result is not None:
            self._save_result()

    def _mouse_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\n已选择坐标 ({x}, {y})")
            try:
                result = self.processor.process((x, y))
                if result is not None:
                    self._show_results(result)
            except Exception as e:
                print(f"处理错误：{str(e)}")
                self.target_coord = None


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description='物体处理工具')
        parser.add_argument('--rgb_img', required=True, help="原始图像路径")
        parser.add_argument('--seg_img', required=True, help="分割图像路径")
        parser.add_argument('--output', required=True, help="输出目录")
        parser.add_argument('--function', required=True, choices=['remove', 'move'],
                            help="功能选择：remove-移除物体，move-移动物体")
        return parser.parse_args()


    args = parse_args()
    processor = DynamicObjectProcessor(
        args.rgb_img,
        args.seg_img,
        args.output,
        args.function
    )
    processor.run()