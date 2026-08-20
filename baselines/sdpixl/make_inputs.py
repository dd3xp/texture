"""生成 SD-piXL baseline 探针实验的输入。

产出两类输入，对应要验证的两个不同问题：

- flat_brown.png   整幅纯棕色。测"给定纯色 + 材质语义，能不能长出纹理"。
- crate_brown.png  纯棕色圆角矩形放在白底上。测边界保持——
                   本任务里 sprite 轮廓是神圣的，纹理不许溢出、轮廓不许糊。

另外写出一个 8 色木质调色板，色相全部落在输入棕色附近，
这样"调色板"这一约束和"输入是纯棕色"这一设定是自洽的。
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


def make_palette(path: Path) -> None:
    path.write_text("\n".join(WOOD_RAMP) + "\n")


def main() -> None:
    out = Path(__file__).parent / "assets"
    out.mkdir(parents=True, exist_ok=True)

    make_flat(out / "flat_brown.png")
    make_crate(out / "crate_brown.png")
    make_palette(out / "wood8.hex")

    for f in sorted(out.iterdir()):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
