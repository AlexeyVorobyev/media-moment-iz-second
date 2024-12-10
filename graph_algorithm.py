import enum
from collections import deque

import cv2
import numpy as np


class ImageMode(enum.Enum):
    GRAYSCALE = 0
    GAUSSIAN = 1
    CONTRAST = 2
    FILTERED = 5


class GraphAlgorithm:
    _image_size: (int, int)
    _deviation: float
    _kernel_size: int
    _contrast: int
    _size_component_threshold: int

    # Сдвиги, при помощи их будем получать соседние пиксели.
    # Сдвиги идут по часовой стрелки, начиная со сдвига для верхнего левого соседа.
    _shifts = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1)
    )

    def __init__(
            self,
            image_size: (int, int) = (500, 500),
            deviation: float = 1,
            kernel_size: int = 5,
            image_show_list: list[ImageMode] = None,
            contrast: int = 10,
            size_component_threshold=100
    ):
        """
        Инициализация класса
        :param image_size: размер изображения
        :param deviation: отклонение
        :param kernel_size: размерность ядра
        :param image_show_list: настройка отображения
        """
        if image_show_list is None:
            self._image_show_list = [
                ImageMode.GRAYSCALE,
                ImageMode.GAUSSIAN,
            ]
        else:
            self._image_show_list = image_show_list
        self._image_size = image_size
        self._deviation = deviation
        self._kernel_size = kernel_size
        self._contrast = contrast
        self._size_component_threshold = size_component_threshold

    def __preprocess_image(self, path_to_image: str) -> np.ndarray:
        """
        Провести предобработку изображения, считать из файла, привести к оттенкам серого, изменить размер
        :param path_to_image:
        :return:
        """
        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self._image_size)

        if ImageMode.GRAYSCALE in self._image_show_list:
            cv2.imshow("GrayScale", img)

        img = cv2.GaussianBlur(img, (self._kernel_size, self._kernel_size), sigmaX=self._deviation,
                               sigmaY=self._deviation)

        if ImageMode.GAUSSIAN in self._image_show_list:
            cv2.imshow("Gaussian", img)

        return img

    # Возвращает матрицу контрастности, полученную из предобработанного изображения
    def _get_matrix_contrast(
            self,
            image: np.array
    ) -> np.array:
        shape = image.shape
        matrix_contrast = np.zeros(shape=shape)
        contrast = np.float16(self._contrast)

        # Пиксель является контрастным, если хотя бы с одним соседним пикселем он контрастирует.
        def _is_contrast_pixel():
            for shift_i, shift_j in self._shifts:
                item1 = int(image[i, j])
                item2 = int(image[i + shift_i, j + shift_j])
                if abs(item1 - item2) >= contrast:
                    return np.uint8(255)
            return np.uint8(0)

        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                matrix_contrast[i, j] = _is_contrast_pixel()

        return matrix_contrast

    # Возвращает список компонент связностей контрастных пикселей, полученный из матрицы контрастности.
    def _get_connectivity_components(slef, matrix_contrast: np.array) -> list[list[tuple[int, int]]]:
        shape = matrix_contrast.shape

        used = np.zeros(
            shape=shape,
            dtype=np.bool_
        )  # used[i, j] - был ли (i, j)-ый пиксель посещён или будет ли посещён.
        components = list()

        # Пиксель является корректным, если его координаты не выходят за диапазоны.
        def is_correct_pixel():
            return 0 <= i < shape[0] and 0 <= j < shape[1]

        for i in range(shape[0]):
            for j in range(shape[1]):
                # Если пиксель контрастный, не был и не будет посещён, то запускаем из него поиск в ширину.
                if matrix_contrast[i, j] == 255 and not used[i, j]:
                    component = [(i, j)]  # Новая компонента связности.
                    queue = deque()  # Очередь пикселей.

                    queue.append((i, j))  # Добавляем стартовый пиксель.
                    used[i, j] = True  # Отмечаем, что он был посещён.

                    # Пока очередь не пуста.
                    while queue:
                        # Извлекаем координаты очередного пикселя из очереди.
                        x, y = queue[0]
                        queue.popleft()

                        # Перебираем его соседей.
                        for shift_x, shift_y in slef._shifts:
                            # Координаты соседа.
                            x_, y_ = x + shift_x, y + shift_y

                            # Если сосед корректный, контрастный и не был посещён, то ...
                            if is_correct_pixel() and matrix_contrast[x_, y_] and not used[x_, y_]:
                                component.append((x_, y_))  # Добавляем соседа в компоненту.
                                queue.append((x_, y_))  # Добавляем соседа в очередь.
                                used[x_, y_] = True  # Отмечаем, что сосед будет посещён.

                    components.append(component)

        return components

    # Возвращает список компонент связностей контрастных пикселей, которые имеют число пикселей >= size_component.
    def _filter_connectivity_components(
            self,
            components: list[list[tuple[int, int]]]
    ) -> list[list[tuple[int, int]]]:
        return [component for component in components if len(component) >= self._size_component_threshold]

    # Возвращает массив контурного изображения, полученного из списка компонент связностей.
    def _get_array_contour_image(
            self,
            components: list[list[tuple[int, int]]],
            shape: tuple[int, int, int]
    ) -> np.array:
        array_image = np.empty(shape=shape, dtype=np.uint8)

        # Делаем все пиксели черными.
        white, black = np.uint8(255), np.uint8(0)
        array_image = np.vectorize(lambda _: black)(array_image)

        # Делаем пиксели из компонент связностей белыми.
        for component in components:
            for i, j in component:
                array_image[i, j] = white

        return array_image

    def process_image_with_return(
            self,
            image_path: str
    ):
        """
        Провести обработку алгоритмом на графах
        :param image_path:
        :return:
        """
        img = self.__preprocess_image(image_path)

        # Получаем матрицу контрастности
        matrix_contrast = self._get_matrix_contrast(img)

        if ImageMode.CONTRAST in self._image_show_list:
            cv2.imshow("Contrast", matrix_contrast)

        # Компоненты связностей контрастных пикселей.
        components = self._get_connectivity_components(matrix_contrast)

        # Фильтруем компоненты связности
        components_filtered = self._filter_connectivity_components(components)

        # Получаем результирющее изображение
        result_img = self._get_array_contour_image(components_filtered, img.shape)

        return result_img
