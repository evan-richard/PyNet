import random


def random_rgb():
    return random.randint(0, 255)


def hex_to_rgb(hexString):
    return [
        round(round(int(hexString.lstrip("#")[i : i + 2], 16) / 2.55, 2) / 100, 4)
        for i in (0, 2, 4)
    ]


def build_theme_id(props):
    return (
        hex_to_rgb(props.get("--foreground-default"))
        + hex_to_rgb(props.get("--foreground-secondary"))
        + hex_to_rgb(props.get("--foreground-tertiary"))
        + hex_to_rgb(props.get("--background-default"))
        + hex_to_rgb(props.get("--background-secondary"))
        + hex_to_rgb(props.get("--background-tertiary"))
        + hex_to_rgb(props.get("--primary-default"))
        + hex_to_rgb(props.get("--primary-dark"))
        + hex_to_rgb(props.get("--primary-light"))
        + hex_to_rgb(props.get("--error-default"))
        + hex_to_rgb(props.get("--error-dark"))
        + hex_to_rgb(props.get("--error-light"))
    )


def run_sample_themes(brain):
    number_of_colors = 12
    rgb_per_color = 3
    sample_size = 10000
    results = []

    for index_of_set in range(0, sample_size):
        input_lst = []
        original_rgb = []
        for index in range(0, number_of_colors * rgb_per_color):
            rgb_val = random_rgb()
            original_rgb.append(rgb_val)
            input_lst.append(round(round(rgb_val / 2.55, 2) / 100, 4))
        result = brain.run(input_lst)
        results.append(dict(theme=original_rgb, score=result))

    return results


def rgb_to_theme(rgb_lst):
    hex_lst = []
    for rgb in rgb_lst:
        rgb = rgb.get("theme")
        hex_lst.append(
            {
                "properties": {
                    "--foreground-default": "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]),
                    "--foreground-secondary": "#%02x%02x%02x"
                    % (rgb[3], rgb[4], rgb[5]),
                    "--foreground-tertiary": "#%02x%02x%02x" % (rgb[6], rgb[7], rgb[8]),
                    "--background-default": "#%02x%02x%02x"
                    % (rgb[9], rgb[10], rgb[11]),
                    "--background-secondary": "#%02x%02x%02x"
                    % (rgb[12], rgb[13], rgb[14]),
                    "--background-tertiary": "#%02x%02x%02x"
                    % (rgb[15], rgb[16], rgb[17]),
                    "--primary-default": "#%02x%02x%02x" % (rgb[18], rgb[19], rgb[20]),
                    "--primary-dark": "#%02x%02x%02x" % (rgb[21], rgb[22], rgb[23]),
                    "--primary-light": "#%02x%02x%02x" % (rgb[24], rgb[25], rgb[26]),
                    "--error-default": "#%02x%02x%02x" % (rgb[27], rgb[28], rgb[29]),
                    "--error-dark": "#%02x%02x%02x" % (rgb[30], rgb[31], rgb[32]),
                    "--error-light": "#%02x%02x%02x" % (rgb[33], rgb[34], rgb[35]),
                }
            }
        )
    return hex_lst
