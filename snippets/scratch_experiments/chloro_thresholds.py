from bigcrittercolor import writeBasicColorMetrics

saturation_min = 0.25
lightness_min = 0.25
saturation_min_blues = 0.2
lightness_min_blues = 0.2

# Define bright color thresholds using triple conditions (hue + lightness + saturation)
bright_color_thresholds = [
    # Red (0-20 and 340-360 degrees, wraps around) - bright
    ("triple",
     ("hls", 0, (340, 20), "between"),  # Hue range in degrees (wraps around)
     ("hls", 1, lightness_min, "above"),  # Lightness > lightness_min
     ("hls", 2, saturation_min, "above"),  # Saturation > saturation_min
     "bright_red"),

    # Orange (20-50 degrees) - bright
    ("triple",
     ("hls", 0, (20, 50), "between"),
     ("hls", 1, lightness_min, "above"),
     ("hls", 2, saturation_min, "above"),
     "bright_orange"),

    # Yellow (50-80 degrees) - bright
    ("triple",
     ("hls", 0, (50, 80), "between"),
     ("hls", 1, lightness_min, "above"),
     ("hls", 2, saturation_min, "above"),
     "bright_yellow"),

    # Green (80-160 degrees) - bright
    ("triple",
     ("hls", 0, (80, 160), "between"),
     ("hls", 1, lightness_min, "above"),
     ("hls", 2, saturation_min, "above"),
     "bright_green"),

    # Cyan (160-200 degrees) - bright
    ("triple",
     ("hls", 0, (160, 200), "between"),
     ("hls", 1, lightness_min_blues, "above"),
     ("hls", 2, saturation_min_blues, "above"),
     "bright_cyan"),

    # Blue (200-260 degrees) - bright
    ("triple",
     ("hls", 0, (200, 260), "between"),
     ("hls", 1, lightness_min_blues, "above"),
     ("hls", 2, saturation_min_blues, "above"),
     "bright_blue"),

    # Magenta/Purple (260-340 degrees) - bright
    ("triple",
     ("hls", 0, (260, 340), "between"),
     ("hls", 1, lightness_min, "above"),
     ("hls", 2, saturation_min, "above"),
     "bright_magenta"),

    # black melanism
     ("hls", 1, 0.2, "below")
]

# Call writeBasicColorMetrics with the bright color thresholds
writeBasicColorMetrics(
    img_ids=None,
    from_stage="segment",
    batch_size=1000,
    get_color_metrics=False,
    get_shape_texture_metrics=False,
    threshold_metrics=bright_color_thresholds,
    pattern_subfolder=None,
    show=True,
    print_steps=True,
    data_folder='D:/bcc/chloros'
)