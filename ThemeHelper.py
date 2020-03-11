def hex_to_reg(hexString):
    return [int(hexString.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]


def build_theme_id(theme):
    props = theme.get('properties')
    return hex_to_reg(props.get('--foreground-default'))\
        + hex_to_reg(props.get('--foreground-secondary'))\
        + hex_to_reg(props.get('--foreground-tertiary'))\
        + hex_to_reg(props.get('--foreground-quaternary'))\
        + hex_to_reg(props.get('--foreground-light'))\
        + hex_to_reg(props.get('--background-default'))\
        + hex_to_reg(props.get('--background-secondary'))\
        + hex_to_reg(props.get('--background-tertiary'))\
        + hex_to_reg(props.get('--background-light'))\
        + hex_to_reg(props.get('--primary-default'))\
        + hex_to_reg(props.get('--primary-dark'))\
        + hex_to_reg(props.get('--primary-light'))\
        + hex_to_reg(props.get('--error-default'))\
        + hex_to_reg(props.get('--error-dark'))\
        + hex_to_reg(props.get('--error-light'))
