use ratatui::prelude::*;

pub const PINK: Color = Color::Rgb(0xFF, 0x5A, 0x45);
pub const GREEN: Color = Color::Rgb(0x6F, 0xCF, 0x97);
pub const RED: Color = Color::Rgb(0xE2, 0x5A, 0x5A);
pub const PURPLE: Color = Color::Rgb(0x6F, 0xA0, 0xD8);
pub const META: Color = Color::Rgb(0xB0, 0xB0, 0xB0);
pub const DIVIDER: Color = Color::Rgb(0x6E, 0x6E, 0x6E);
pub const TEXT: Color = Color::Rgb(0xFF, 0xFF, 0xFF);
pub const GOLD: Color = Color::Rgb(0xE8, 0xB8, 0x4E);

pub const SPINNER: &[char] = &['⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub fn text_style() -> Style {
    Style::default().fg(TEXT)
}

pub fn meta_style() -> Style {
    Style::default().fg(META)
}

pub fn divider_style() -> Style {
    Style::default().fg(DIVIDER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_colors_are_distinct() {
        let colors = [PINK, GREEN, RED, PURPLE, META, DIVIDER, TEXT];
        for (i, a) in colors.iter().enumerate() {
            for b in colors.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn spinner_is_non_empty() {
        assert!(!SPINNER.is_empty());
    }
}
