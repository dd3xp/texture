"""生成 SD-piXL baseline 探针实验的输入。

产出两类输入，对应要验证的两个不同问题：

- flat_brown.png   整幅纯棕色。测"给定纯色 + 材质语义，能不能长出纹理"。
- crate_brown.png  纯棕色圆角矩形放在白底上。测边界保持——
                   本任务里 sprite 轮廓是神圣的，纹理不许溢出、轮廓不许糊。

另外写出两个调色板：

- wood8.hex   8 色木质色阶，色相全部落在输入棕色附近。
- wood9bg.hex 同上再加一个纯白，专门给背景用。

wood8 是最初的设计，但它有缺陷：调色板里没有中性色，
crate 输入的白色背景会被就近映射成最浅的棕，于是"背景"和"物体"变成同色系，
纹理漫进背景时看不出来。wood9bg 给背景留了专用色，
这样纹理一旦越界就是明确可判的。
"""

from pathlib import Path

from PIL import Image, ImageDraw

# 基色与派生的木质色阶（暗 -> 亮）。基色取 index 4。
BASE_BROWN = "#8b5a2b"
WOOD_RAMP = [
    "2b1a10",
    "3d2617",
    "54341f",
    "6b4429",
    "8b5a2b",  # BASE_BROWN
    "a67340",
    "c08d55",
    "d9a970",
]

SIZE = 512  # 源图分辨率；输出像素预算由 --size 单独控制


def make_flat(path: Path) -> None:
    Image.new("RGB", (SIZE, SIZE), BASE_BROWN).save(path)


def make_crate(path: Path) -> None:
    """纯棕色圆角矩形置于白底，用来测边界。"""
    img = Image.new("RGB", (SIZE, SIZE), "#ffffff")
    draw = ImageDraw.Draw(img)
    margin = SIZE // 8
    draw.rounded_rectangle(
        [margin, margin, SIZE - margin, SIZE - margin],
        radius=SIZE // 32,
        fill=BASE_BROWN,
    )
    img.save(path)


BG_WHITE = "ffffff"


def make_palette(path: Path, colors: list[str]) -> None:
    path.write_text("\n".join(colors) + "\n")


def main() -> None:
    out = Path(__file__).parent / "assets"
    out.mkdir(parents=True, exist_ok=True)

    make_flat(out / "flat_brown.png")
    make_crate(out / "crate_brown.png")
    make_palette(out / "wood8.hex", WOOD_RAMP)
    make_palette(out / "wood9bg.hex", WOOD_RAMP + [BG_WHITE])

    for f in sorted(out.iterdir()):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
