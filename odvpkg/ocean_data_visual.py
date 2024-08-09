from abc import ABC, abstractmethod
import glob
import os
import shutil
from typing import Any
import imageio
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

# base module: draw fun -> image generator(colorbar) -> joining image generator(multi image joiner)

class ColorbarMode(ABC):
    @abstractmethod
    def _add(self, fig: Figure, image: Any, ax: Any):
        pass

    def __call__(self, fig: Figure, image: Any, ax: Any) -> None:
        if image:
            self._add(fig, image, ax)

class NoneColorbar(ColorbarMode):
    def _add(self, fig: Figure, image: Any, ax: Any):
        pass

class DrawFun(ABC):
    @abstractmethod
    def __call__(self, ax: Axes, frame: np.ndarray|list[np.ndarray], vmin: float, vmax: float):
        pass

class DrawNoFrame(DrawFun):
    def __call__(self, ax: Axes, frame: np.ndarray|list[np.ndarray], vmin: float, vmax: float):
        pass

class ImageGenerator(ABC):
    def __init__(self, figsize: tuple[int], draw_fun: DrawFun|list[DrawFun], fontsize: int, 
                image_format: str, colorbar_mode: ColorbarMode) -> None:
        self._figsize = figsize
        self._draw_fun = draw_fun
        self._fontsize = fontsize
        self._image_format = image_format
        self._colorbar_mode = colorbar_mode

    def set_draw_fun(self, draw_fun: DrawFun|list[DrawFun]):
        self._draw_fun = draw_fun

    @abstractmethod
    def _draw(self, data: Any, image_folder: str, image_name: str, title: str, vmin: float, vmax: float) -> None:
        pass

    def __call__(self, data: Any, 
            image_folder: str, image_name: str, title: str=None, 
            vmin: float=None, vmax: float=None):
        os.makedirs(image_folder, exist_ok=True)
        plt.rcParams.update({"font.size": self._fontsize})
        self._draw(data, image_folder, image_name, title, vmin, vmax)
        plt.cla()
        plt.close()

class NoneImageGenerator(ImageGenerator):
    def __init__(self) -> None:
        super().__init__(figsize=(30, 30), draw_fun=DrawNoFrame(), fontsize=0, image_format="jpg", colorbar_mode=NoneColorbar())

    def _draw(self, data: Any, image_folder: str, image_name: str, title: str, vmin: float, vmax: float) -> None:
        pass

    def __call__(self, data: Any, 
            image_folder: str, image_name: str, title: str=None, 
            vmin: float=None, vmax: float=None):
        pass

class MultiImageJoiner(ABC):
    @abstractmethod
    def __call__(self, out_file_folder: str, out_file_name: str, frame_rate: int, image_folder: str, image_format: str) -> None:
        pass

class MultiImageToNone(MultiImageJoiner):
    def __call__(self, out_file_folder: str, out_file_name: str, frame_rate: int, image_folder: str, image_format: str) -> None:
        pass

class JoiningImageGenerator:
    def __init__(self, image_generator: ImageGenerator, multi_image_joiner: MultiImageJoiner, frame_rate: int, image_format: str) -> None:
        self._image_generator = image_generator
        self._multi_image_joiner = multi_image_joiner
        self._frame_rate = frame_rate
        self._image_format = image_format

    def set_image_generator(self, image_generator: ImageGenerator):
        self._image_generator = image_generator

    def _clear_image_folder(self, image_folder: str):
        shutil.rmtree(image_folder, ignore_errors=True)
        os.makedirs(image_folder)

    def __call__(self, data: Any, file_folder: str, file_name: str, vmin: float=None, vmax: float=None):
        image_folder = os.path.join(file_folder, f"image_{file_name}")
        self._clear_image_folder(image_folder)
        for t_id in tqdm(range(len(data))):
            image_id_string = str(t_id).rjust(len(str(len(data))), "0")
            image_name = f"image_{file_name}_{image_id_string}"
            self._image_generator(data[t_id], image_folder, image_name, vmin=vmin, vmax=vmax, title=f"t={t_id}T")
        self._multi_image_joiner(file_folder, file_name, self._frame_rate, image_folder, self._image_format)

# colorbar

class HorizontalColorbar(ColorbarMode):
    def __init__(self, aspect: float) -> None:
        self._aspect = aspect

    def _add(self, fig: Figure, image, ax: Any) -> None:
        fig.colorbar(image, ax=ax, orientation="horizontal", aspect=self._aspect)

class VerticalColorbar(ColorbarMode):
    def __init__(self, aspect: float) -> None:
        self._aspect = aspect

    def _add(self, fig: Figure, image, ax: Any) -> None:
        fig.colorbar(image, ax=ax, orientation="vertical", aspect=self._aspect)

# draw fun for hcc: heat, courtourf and courtour(levels generator), inlclude extent generator and frame adder
# template inlucde three layer structure

class ContourLevelsGenerator:
    def __init__(self, level_num: int) -> None:
        self.level_num = level_num

    def __call__(self, vmin: float, vmax: float) -> np.ndarray|int:
        if vmin == None or vmax == None:
            return self.level_num
        if vmin == vmax:
            return self.level_num
        return np.linspace(vmin, vmax, self.level_num)

class HccExtentGenerator:
    @abstractmethod
    def __call__(self, frame: np.ndarray) -> list[int]:
        pass

class NoneExtentGenerator:
    def __call__(self, frame: np.ndarray) -> list[int]:
        return [0, 0, 0, 0]

class HeatExtentGenerator(HccExtentGenerator):
    def __call__(self, frame: np.ndarray) -> list[int]:
        extent = [0, frame.shape[1], 0, frame.shape[0]]
        return extent

class ContourExtentGenerator(HccExtentGenerator):
    def __call__(self, frame: np.ndarray) -> list[int]:
        extent = [0, frame.shape[1], frame.shape[0], 0]
        return extent
    
class FrameAdder:
    def __init__(self, image_alpha: float, extent_generator: HccExtentGenerator) -> None:
        self._image_alpha = image_alpha
        self._extent_generator = extent_generator

    @abstractmethod
    def __call__(self, frame: np.ndarray, ax: Axes, vmin: float, vmax: float):
        pass

class NoneFrameAdder(FrameAdder):
    def __init__(self) -> None:
        super().__init__(0.0, NoneExtentGenerator())

    def __call__(self, frame: np.ndarray, ax: Axes, vmin: float, vmax: float):
        return None

class HeatFrameAdder(FrameAdder):
    def __init__(self, image_alpha: float, aspect: float, cmap: str, interpolation: str) -> None:
        extent_generator = HeatExtentGenerator()
        super().__init__(image_alpha, extent_generator)
        self._aspect = aspect
        self._interpolation = interpolation
        self._cmap = cmap

    def __call__(self, frame: np.ndarray, ax: Axes, vmin: float, vmax: float):
        extent = self._extent_generator(frame)
        image = ax.imshow(frame, cmap=self._cmap, interpolation=self._interpolation,
            alpha=self._image_alpha, aspect=self._aspect, extent=extent,
            vmin=vmin, vmax=vmax)
        if self._image_alpha:
            return image
        else:
            return None

class NoneHeatFrameAdder(HeatFrameAdder):
    def __init__(self, aspect: float) -> None:
        extent_generator = HeatExtentGenerator()
        super().__init__(0.0, extent_generator, None, None)
        self._aspect = aspect
        self._interpolation = None
        self._cmap = None

class ContourfFrameAdder(FrameAdder):
    def __init__(self, image_alpha: float, level_num: int, cmap: str) -> None:
        extent_generator = ContourExtentGenerator()
        super().__init__(image_alpha, extent_generator)
        self._levels_generator = ContourLevelsGenerator(level_num)
        self._cmap = cmap

    def __call__(self, frame: np.ndarray, ax: Axes, vmin: float, vmax: float):
        if self._image_alpha:
            extent = self._extent_generator(frame)
            levels = self._levels_generator(vmin, vmax)
            image = ax.contourf(frame, cmap=self._cmap, levels=levels, extent=extent, alpha=self._image_alpha)
            return image
        else:
            return None

class ContourFrameAdder(FrameAdder):
    def __init__(self, image_alpha: float, color: str, level_num: int) -> None:
        extent_generator = ContourExtentGenerator()
        super().__init__(image_alpha, extent_generator)
        self._levels_generator = ContourLevelsGenerator(level_num)
        self._color = color
        self._level_num = level_num

    def __call__(self, frame: np.ndarray, ax: Axes, vmin: float, vmax: float) -> None:
        if self._image_alpha:
            extent = self._extent_generator(frame)
            levels = self._levels_generator(vmin, vmax)
            ax.contour(frame, colors=self._color, levels=levels, extent=extent, alpha=self._image_alpha)
        return None

class DrawHccFrameTemplate(DrawFun):
    def __init__(self, heat_image_adder: HeatFrameAdder, contourf_image_adder: ContourfFrameAdder, 
            contour_image_adder: ContourFrameAdder) -> None:
        self._heat_image_adder = heat_image_adder
        self._contourf_image_adder = contourf_image_adder
        self._contour_image_adder = contour_image_adder
    
    def __call__(self, ax: Axes, frame: np.ndarray, vmin: float, vmax: float):
        heat_image = self._heat_image_adder(frame, ax, vmin, vmax)
        contourf_image = self._contourf_image_adder(frame, ax, vmin, vmax)
        self._contour_image_adder(frame, ax, vmin, vmax)
        image = heat_image or contourf_image
        return image
    
# HCC: heat, courtourf and courtour, composition

class DrawHcFrame(DrawHccFrameTemplate):
    # HC: heat and contour
    def __init__(self, level_num: int, aspect: float=1, cmap: str=None, heat_interpolation: str=None, 
            contour_color: str="black"):
        heat_image_adder = HeatFrameAdder(1.0, aspect, cmap, heat_interpolation)
        contourf_image_adder = NoneFrameAdder()
        contour_image_adder = ContourFrameAdder(1.0, contour_color, level_num)
        super().__init__(heat_image_adder, contourf_image_adder, contour_image_adder)
        
class DrawHeatFrame(DrawHccFrameTemplate):
    def __init__(self, aspect: float=1, cmap: str=None, heat_interpolation: str=None):
        heat_image_adder = HeatFrameAdder(1.0, aspect, cmap, heat_interpolation)
        contourf_image_adder = NoneFrameAdder()
        contour_image_adder = NoneFrameAdder()
        super().__init__(heat_image_adder, contourf_image_adder, contour_image_adder)

class DrawCcFrame(DrawHccFrameTemplate):
    # CC: contourf and contour
    def __init__(self, level_num: int, aspect: float=1, cmap: str=None, 
            contour_color: str="black"):
        heat_image_adder = NoneHeatFrameAdder(aspect)
        contourf_image_adder = ContourfFrameAdder(1.0, level_num, cmap)
        contour_image_adder = ContourFrameAdder(1.0, contour_color, level_num)
        super().__init__(heat_image_adder, contourf_image_adder, contour_image_adder)

class DrawContourfFrame(DrawHccFrameTemplate):
    def __init__(self, level_num: int, aspect: float=1, cmap: str=None):
        heat_image_adder = NoneHeatFrameAdder(aspect)
        contourf_image_adder = ContourfFrameAdder(1.0, level_num, cmap)
        contour_image_adder = NoneFrameAdder()
        super().__init__(heat_image_adder, contourf_image_adder, contour_image_adder)

class XyGenerator(ABC):
    def __call__(self, data: np.ndarray) -> tuple[np.ndarray]:
        pass

class UshapeXyGenerator:
    def __call__(self, data: np.ndarray)-> tuple[np.ndarray]:
        y_len, x_len = data.shape
        y, x = np.meshgrid(np.arange(y_len), np.arange(x_len), indexing="ij")
        return x, y

class StreamLineFrameAdder(FrameAdder):
    def __init__(self, image_alpha: float, xy_generator: XyGenerator) -> None:
        super().__init__(image_alpha, NoneExtentGenerator())
        self._xy_generator = xy_generator

    def __call__(self, frame: np.ndarray, ax: Axes, vmin: float, vmax: float):
        u, v = frame
        x, y = self._xy_generator(u)
        ax.streamplot(x, y, u, v)
        return None

class DrawStreamPlotFrame(DrawFun):
    def __init__(self, aspect: float=1) -> None:
        self._background_image_adder = NoneHeatFrameAdder(aspect)
        self._stream_plot_image_adder = StreamLineFrameAdder(1, UshapeXyGenerator())
    
    def __call__(self, ax: Axes, frame: np.ndarray, vmin: float, vmax: float):
        self._background_image_adder(frame[0], ax, vmin, vmax)
        self._stream_plot_image_adder(frame, ax, vmin, vmax)
        return None

# plot

class DrawPlotFrame(DrawFun):    
    def __call__(self, ax: Axes, frame: np.ndarray, vmin: float, vmax: float):
        image = ax.plot(frame)
        return image

# multi image joiner and joining image generator

class MultiImageToMovie(MultiImageJoiner):
    def __init__(self, movie_format="mp4") -> None:
        self._movie_format = movie_format

    def __call__(self, out_file_folder: str, out_file_name: str, frame_rate: int, image_folder: str, image_format: str) -> None:
        out_file_path = os.path.join(out_file_folder, out_file_name, self._movie_format)
        cmd = f"rm -rf {out_file_path}"
        os.system(cmd)
        cmd = (
            f"ffmpeg"
            f" -framerate {frame_rate}" 
            f" -pattern_type glob"
            f" -i \"{image_folder}\""
            f" -c:v libx264"
            f" -r 30"
            f" -pix_fmt yuv420p"
            f" {os.path.join(out_file_folder, out_file_name)}"
        )
        os.system(cmd)
        print(f"saving movie to {out_file_path}")

class MultiImageToGif(MultiImageJoiner):
    def __call__(self, out_file_folder: str, out_file_name: str, frame_rate: int, image_folder: str, image_format: str) -> None:
        frame_paths = os.path.join(image_folder, image_format)
        gif_images = []
        frames = glob.glob(frame_paths)
        for frame in frames:
            gif_images.append(imageio.imread(frame))
        imageio.mimsave(os.path.join(out_file_folder, out_file_name), gif_images, duration=1/frame_rate)
        print(f"saving movie to {os.path.join(out_file_folder, out_file_name)}")

# info image generator

class InfoImageGenerator(ImageGenerator):
    def __init__(self, figsize: tuple[int], draw_fun: DrawFun, fontsize: int=0, 
        colorbar_mode: ColorbarMode=NoneColorbar(), image_format: str="jpg") -> None:
        super().__init__(figsize, draw_fun, fontsize, image_format, colorbar_mode)

    def _draw(self, data: Any, image_folder: str, image_name: str, title: str, vmin: float, vmax: float) -> None:
        fig, ax = plt.subplots(figsize=self._figsize)
        image = self._draw_fun(ax, data, vmin, vmax)
        ax.set_title(title, fontsize=self._fontsize)
        self._colorbar_mode(fig, image, ax)
        fig.savefig(os.path.join(image_folder, f"{image_name}.{self._image_format}"))

# purge image generator(axes clearer)

class AxesClearer:
    def __call__(self, ax: Axes) -> None:
        for pos in ["top", "right", "bottom", "left"]:
            ax.spines[pos].set_visible(False)
        ax.axis("off")

class PurseImageGenerator(ImageGenerator):
    def __init__(self, figsize: tuple[int], draw_fun: DrawFun, image_format: str="svg"):
        super().__init__(figsize, draw_fun, 0, image_format, NoneColorbar())
        self._axes_clearer = AxesClearer()

    def _draw(self, data: Any, image_folder: str, image_name: str, title: str, vmin: float, vmax: float) -> None:
        fig, ax = plt.subplots(figsize=self._figsize)
        self._axes_clearer(ax)
        self._draw_fun(ax, data, vmin, vmax)
        fig.savefig(os.path.join(image_folder, f"{image_name}.{self._image_format}"), bbox_inches="tight", pad_inches=0)

# comparing image generator(row col axes manager(row col axes array generator), frame name array generator)

class FrameNameArrayGenerator:
    def __call__(self, frame_names: list[str], frame_num: int) -> Any:
        if not frame_names:
            return [str(frame_index) for frame_index in range(frame_num)]
        if not len(frame_names) == frame_num:
            raise f"the num of frame_names must be same as the frame_num"
        if not len(frame_names) == len(set(frame_names)):
            raise f"frame_names shouldn't contain duplicate name"
        return frame_names

class RowColAxArrayGenerator:
    def __init__(self, ncols: int, nrows: int) -> None:
        self._generate_fun = self._get_generate_fun(ncols, nrows)

    def _get_generate_fun(self, ncols: int, nrows: int):
        if nrows == 1 and ncols == 1:
            fun = lambda axes: [[axes]]
        elif nrows == 1 and ncols > 1:
            fun = lambda axes: [axes]
        elif nrows > 1 and ncols == 1:
            fun = lambda axes: [[ax] for ax in axes]
        else:
            fun = lambda axes: axes
        return fun
    
    def __call__(self, axex) -> list[list[Axes]]:
        return self._generate_fun(axex)

class RowColAxesManager:
    def __init__(self, frame_num: int, ncols: int) -> None:
        nrows = int(frame_num / ncols)
        self._rc_ax_array_generator = RowColAxArrayGenerator(ncols, nrows)
        self._nrows = nrows
        self._ncols = ncols
        self._frame_num = frame_num

    def get_nrows(self):
        return self._nrows

    def bind_axex(self, axex):
        self._ax_array = self._rc_ax_array_generator(axex)
        self.axex = axex

    def clear(self):
        self._ax_array = None
        self.axex = None

    def __len__(self):
        return self._frame_num

    def __getitem__(self, index: int) -> Axes:
        ax = self._ax_array[int(index % self._nrows)][index // self._nrows]
        return ax

class CompImageGenerator(ImageGenerator):
    _frame_name_array_generator = FrameNameArrayGenerator()

    def __init__(self, figsize: tuple[int], frame_num: int, draw_fun: DrawFun|list[DrawFun],
            fontsize: int=0, image_format: str="jpg", colorbar_mode: ColorbarMode=NoneColorbar(), 
            ncols: int=1, frame_names: list[str]=None, 
            purse_image_size: int=None, purse_image_format: str="svg", 
            ) -> None:
        super().__init__(figsize, draw_fun, fontsize, image_format, colorbar_mode)
        self._frame_num = frame_num
        self._frame_names = self._frame_name_array_generator(frame_names, frame_num)
        self._purse_image_generator = self._create_purse_image_generator(purse_image_size, purse_image_format)
        self._rc_axex_manager = RowColAxesManager(frame_num, ncols)
        self._nrows = self._rc_axex_manager.get_nrows()
        self._ncols = ncols

    def _draw(self, data, image_folder: str, image_name: str, title: str, vmin: float, vmax: float) -> None:
        fig, axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=self._figsize)
        self._rc_axex_manager.bind_axex(axes)

        for frame_index, frame in enumerate(data):
            draw_fun = self._draw_fun[frame_index]
            frame_name = self._frame_names[frame_index]
            # purse_image
            self._purse_image_generator.set_draw_fun(draw_fun)
            purse_image_folder = os.path.join(image_folder, image_name)
            self._purse_image_generator(frame, purse_image_folder, frame_name, vmin=vmin, vmax=vmax)
            # frame
            ax = self._rc_axex_manager[frame_index]
            image = draw_fun(ax, frame, vmin, vmax)
            ax.set_title(label=frame_name)
        # colorbar
        self._colorbar_mode(fig, image, self._rc_axex_manager.axex)
        # save
        fig.savefig(os.path.join(image_folder, image_name))
        self._rc_axex_manager.clear()
    
    def _create_purse_image_generator(self, image_size, image_format):
        if image_size:
            return PurseImageGenerator(image_size, image_format)
        else:
            return NoneImageGenerator()
        
    def set_draw_fun(self, draw_fun: DrawFun|list[DrawFun]):
        if len(draw_fun) == self._frame_num:
            self._draw_fun = draw_fun
        elif len(draw_fun) == 1:
            self._draw_fun = [draw_fun] * self._frame_num
        else:
            raise f"the num of draw_funs must be same as the frame_num"

def main():
    draw_fun = DrawCcFrame(level_num=5, aspect=1)
    image_generator = CompImageGenerator((70, 130), 3, [draw_fun, draw_fun, draw_fun], fontsize=80, 
            image_format="jpg", colorbar_mode=HorizontalColorbar(40), ncols=3)
    data = [np.random.rand(30, 30), np.random.rand(30, 30), np.random.rand(30, 30)]
    image_generator(data, "image", "tem", title="tem")

if __name__ == "__main__":
    main()
