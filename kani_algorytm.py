import enum
import math
from typing import Callable

import cv2
import numpy as np

from matrix_operators import MatrixOperator, SobelOperator


class ImageShowKaniAlgorythmEnum(enum.Enum):
    GRAYSCALE = 0
    GAUSSIAN = 1
    GRAD_LENGTH = 2
    GRAD_ANGLE = 3
    SUPPRESSED = 4
    FILTERED = 5

class KaniAlgorythm:

    _image_size: (int, int)
    _deviation: float
    _kernel_size: int
    _image_show_list: list[ImageShowKaniAlgorythmEnum]
    _matrix_operator: MatrixOperator
    _threshold_dividers: (int, int)

    def __init__(
            self,
            image_size: (int, int) = (500, 500),
            deviation: float = 1,
            kernel_size: int = 5,
            image_show_list: list[ImageShowKaniAlgorythmEnum] =None,
            threshold_dividers: (int, int) = (5, 5)
    ):
        """
        Инициализация класса
        :param image_size: размер изображения
        :param deviation: отклонение
        :param kernel_size: размерность ядра
        :param image_size: размер изображения
        :param image_show_list: настройка отображения
        """
        if image_show_list is None:
            self._image_show_list = [
                ImageShowKaniAlgorythmEnum.GRAYSCALE,
                ImageShowKaniAlgorythmEnum.GAUSSIAN,
            ]
        else:
            self._image_show_list = image_show_list
        self._image_size = image_size
        self._deviation = deviation
        self._kernel_size = kernel_size
        self._matrix_operator = SobelOperator()
        self._threshold_dividers = threshold_dividers

    def __preprocess_image(self, path_to_image: str) -> np.ndarray:
        """
        Провести предобработку изображения, считать из файла, привести к оттенкам серого, изменить размер
        :param path_to_image:
        :return:
        """
        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self._image_size)

        if ImageShowKaniAlgorythmEnum.GRAYSCALE in self._image_show_list:
            cv2.imshow("GrayScale", img)

        img = cv2.GaussianBlur(img, (self._kernel_size, self._kernel_size), sigmaX=self._deviation, sigmaY=self._deviation)

        if ImageShowKaniAlgorythmEnum.GAUSSIAN in self._image_show_list:
            cv2.imshow("Gaussian", img)

        return img

    def __get_gradients(
            self,
            img: np.ndarray,
            g_x_matrix_operator: Callable[[np.ndarray, int, int], int],
            g_y_matrix_operator: Callable[[np.ndarray, int, int], int]
    ) -> list[list[tuple]]:
        """
        Получить матрицу градиентов для пикселей изображения
        :param img:
        :param g_x_matrix_operator:
        :param g_y_matrix_operator:
        :return:
        """
        gradients = []
        for x in range(1, img.shape[0] - 1):
            gradient_row = []
            for y in range(1, img.shape[1] - 1):
                Gx = g_x_matrix_operator(img, x, y)
                Gy = g_y_matrix_operator(img, x, y)
                gradient_row.append((Gx, Gy))
            gradients.append(gradient_row)
        return gradients

    def __get_grad_length(self, img: np.ndarray, grads: list[list[tuple]]) -> np.ndarray:
        """
        Получить матрицу длин градиентов
        :param img:
        :param grads:
        :return:
        """
        grads_length = np.zeros((img.shape[0], img.shape[1]))
        grad_x_coord = 0
        for x in range(1, img.shape[0] - 1):
            grad_y_coord = 0
            for y in range(1, img.shape[1] - 1):
                Gx, Gy = grads[grad_x_coord][grad_y_coord]
                grads_length[x, y] = math.sqrt(Gx ** 2 + Gy ** 2)
                grad_y_coord = grad_y_coord + 1
            grad_x_coord = grad_x_coord + 1
        return grads_length

    def __get_corner_by_grad(self, grad: tuple) -> int:
        """
        Получить округлённое значение угла по его градиенту
        :param grad:
        :return:
        """
        Gx, Gy = grad
        tang = Gy / Gx if Gx != 0 else 999
        if Gx > 0 > Gy and tang < -2.414 or Gx < 0 and Gy < 0 and tang > 2.414:
            return 0
        elif Gx > 0 > Gy and tang < -0.414:
            return 1
        elif Gx > 0 > Gy and tang > -0.414 or Gx > 0 and Gy > 0 and tang < 0.414:
            return 2
        elif Gx > 0 and Gy > 0 and tang < 2.414:
            return 3
        elif Gx > 0 and Gy > 0 and tang > 2.414 or Gx < 0 < Gy and tang < -2.414:
            return 4
        elif Gx < 0 < Gy and tang < -0.414:
            return 5
        elif Gx < 0 < Gy and tang > -0.414 or Gx < 0 and Gy < 0 and tang < 0.414:
            return 6
        elif Gx < 0 and Gy < 0 and tang < 2.414:
            return 7
        if Gx == 0:
            if Gy > 0:
                return 4
            elif Gy <= 0:
                return 0
        else:
            if Gy > 0:
                return 2
            elif Gy <= 0:
                return 6

    def __get_corners(self, img: np.ndarray, grads: list[list[tuple]]) -> np.ndarray:
        """
        Получить матрицу углов градиентов
        :param img:
        :param grads:
        :return:
        """
        corners = np.zeros((img.shape[0], img.shape[1]))
        grads_len = len(grads[0])
        corner_x = 1
        for i in range(len(grads)):
            corner_y = 1
            for j in range(grads_len):
                corners[corner_x, corner_y] = self.__get_corner_by_grad(grads[i][j])
                corner_y += 1
            corner_x += 1
        return corners

    def __not_max_suppress(self, grads_len: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Подавление немаксимумов
        :param grads_len: Матрица длин градиентов
        :param corners: Матрица углов градиентов
        :return:
        """
        height, width = grads_len.shape
        bordered_image = np.zeros_like(grads_len)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                angle = int(corners[x][y])
                first_neigh, second_neigh = self.__get_grad_neighbors_by_angle(grads_len, x, y, angle)

                if grads_len[x][y] > first_neigh and grads_len[x][y] > second_neigh:
                    bordered_image[x][y] = 255
                else:
                    bordered_image[x][y] = 0

        return bordered_image

    def __get_grad_neighbors_by_angle(self, grads_len: np.ndarray, x: int, y: int, angle: int) -> tuple:
        """
        Получить длины градиентов двух соседних пикселей
        :param grads_len:
        :param x:
        :param y:
        :param angle:
        :return:
        """
        if angle == 0 or angle == 4:
            return grads_len[x + 1][y], grads_len[x - 1][y]
        elif angle == 1 or angle == 5:
            return grads_len[x - 1][y + 1], grads_len[x + 1][y - 1]
        elif angle == 2 or angle == 6:
            return grads_len[x][y + 1], grads_len[x][y - 1]
        elif angle == 3 or angle == 7:
            return grads_len[x + 1][y + 1], grads_len[x - 1][y - 1]
        else:
            return -9999999, -9999999

    def __double_threshold_filter(
            self,
            img: np.ndarray,
            img_with_borders: np.ndarray,
            grads_len: np.ndarray,
    ):
        """
        Выполнить двойную пороговую фильтрацию
        :param img: Изображение
        :param img_with_borders: Изображение с уже отмеченными границами
        :param grads_len: Матрица длин градиентов
        :return:
        """
        max_gradient = np.max(grads_len)
        print(max_gradient)
        lower_bound = max_gradient / self._threshold_dividers[0]
        upper_bound = max_gradient / self._threshold_dividers[1]
        print(f'Нижняя граница {lower_bound}, Верхняя: {upper_bound}')
        filtered_img = np.zeros(img.shape)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                gradient = grads_len[i][j]
                if img_with_borders[i][j] == 255:
                    # Если выше верхней границы, то точно входит
                    if gradient > upper_bound:
                        filtered_img[i][j] = 255
                    elif lower_bound <= gradient <= upper_bound:  # Если между двумя границами - нужно проверить соседей
                        has_neigh_border = False
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                if (
                                        img_with_borders[i + k][j + l] == 255
                                        and img_with_borders[i + k][j + l] >= upper_bound
                                ):
                                    has_neigh_border = True
                        if has_neigh_border:
                            img_with_borders[i][j] = 255
        return filtered_img

    def process_image(
            self,
            image_path: str
    ):
        """
        Провести обработку алгоритмом Канни
        :param image_path:
        :return:
        """
        img = self.__preprocess_image(image_path)

        gradients = self.__get_gradients(img, self._matrix_operator.x_matrix, self._matrix_operator.y_matrix)

        grads_lengths = self.__get_grad_length(img, grads=gradients)

        if ImageShowKaniAlgorythmEnum.GRAD_LENGTH in self._image_show_list:
            cv2.imshow("Grad lengths", cv2.resize(grads_lengths, self._image_size))
            print('Матрица значений длин градиентов:')
            print(grads_lengths)

        corners = self.__get_corners(img, gradients)

        if ImageShowKaniAlgorythmEnum.GRAD_ANGLE in self._image_show_list:
            cv2.imshow("Grad angles", cv2.resize(corners, self._image_size))
            print('Матрица значений углов градиентов:')
            print(corners)

        suppressed_img = self.__not_max_suppress(grads_lengths, corners)

        if ImageShowKaniAlgorythmEnum.SUPPRESSED in self._image_show_list:
            cv2.imshow("Suppressed", suppressed_img)

        result_img = self.__double_threshold_filter(
            img,
            suppressed_img,
            grads_lengths
        )

        if ImageShowKaniAlgorythmEnum.FILTERED in self._image_show_list:
            cv2.imshow("Filtered", result_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image_with_return(
            self,
            image_path: str
    ):
        """
        Провести обработку алгоритмом Канни
        :param image_path:
        :return:
        """
        img = self.__preprocess_image(image_path)

        gradients = self.__get_gradients(img, self._matrix_operator.x_matrix, self._matrix_operator.y_matrix)

        grads_lengths = self.__get_grad_length(img, grads=gradients)

        if ImageShowKaniAlgorythmEnum.GRAD_LENGTH in self._image_show_list:
            cv2.imshow("Grad lengths", cv2.resize(grads_lengths, self._image_size))
            print('Матрица значений длин градиентов:')
            print(grads_lengths)

        corners = self.__get_corners(img, gradients)

        if ImageShowKaniAlgorythmEnum.GRAD_ANGLE in self._image_show_list:
            cv2.imshow("Grad angles", cv2.resize(corners, self._image_size))
            print('Матрица значений углов градиентов:')
            print(corners)

        suppressed_img = self.__not_max_suppress(grads_lengths, corners)

        if ImageShowKaniAlgorythmEnum.SUPPRESSED in self._image_show_list:
            cv2.imshow("Suppressed", suppressed_img)

        max_gradient = np.max(grads_lengths)
        print(max_gradient)
        lower_bound = max_gradient / self._threshold_dividers[0]
        upper_bound = max_gradient / self._threshold_dividers[1]

        result_img = self.__double_threshold_filter(
            img,
            suppressed_img,
            grads_lengths
        )

        return result_img, (lower_bound, upper_bound)
