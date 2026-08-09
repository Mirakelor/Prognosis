pub fn extract_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&s[start..=i]);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

pub fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != 0x1b {
            let start = i;
            while i < bytes.len() && bytes[i] != 0x1b {
                i += 1;
            }
            out.push_str(&text[start..i]);
            continue;
        }
        i += 1;
        match bytes.get(i) {
            None => break,
            Some(b'[') => {
                i += 1;
                while i < bytes.len() && !(0x40..=0x7e).contains(&bytes[i]) {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
            }
            Some(b']') => {
                i += 1;
                while i < bytes.len() && bytes[i] != 0x07 {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
            }
            Some(_) => {
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_object() {
        assert_eq!(extract_json_object(r#"{"a": 1}"#), Some(r#"{"a": 1}"#));
    }

    #[test]
    fn nested_objects() {
        assert_eq!(
            extract_json_object(r#"{"a": {"b": 2}}"#),
            Some(r#"{"a": {"b": 2}}"#)
        );
    }

    #[test]
    fn braces_inside_strings_are_ignored() {
        assert_eq!(
            extract_json_object(r##"{"reaction": "他说了}"}"##),
            Some(r##"{"reaction": "他说了}"}"##)
        );
        assert_eq!(
            extract_json_object(r#"{"a": "x{y", "b": 1}"#),
            Some(r#"{"a": "x{y", "b": 1}"#)
        );
    }

    #[test]
    fn escaped_quotes_inside_strings() {
        assert_eq!(
            extract_json_object(r#"{"a": "say \"hi\""}"#),
            Some(r#"{"a": "say \"hi\""}"#)
        );
    }

    #[test]
    fn first_object_when_multiple() {
        assert_eq!(
            extract_json_object(r#"explanation {"error": 0.5} then {"error": 1.0}"#),
            Some(r#"{"error": 0.5}"#)
        );
    }

    #[test]
    fn text_without_object_returns_none() {
        assert_eq!(extract_json_object("no json here"), None);
    }

    #[test]
    fn unterminated_object_returns_none() {
        assert_eq!(extract_json_object(r#"{"a": 1"#), None);
    }

    #[test]
    fn strip_ansi_removes_csi_sequences() {
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m"), "red");
        assert_eq!(strip_ansi("a\x1b[1;32mb\x1b[mc"), "abc");
    }

    #[test]
    fn strip_ansi_removes_osc_and_single_escape() {
        assert_eq!(strip_ansi("\x1b]0;title\x07body"), "body");
        assert_eq!(strip_ansi("x\x1b7y\x1b8z"), "xyz");
    }

    #[test]
    fn strip_ansi_keeps_plain_text() {
        assert_eq!(strip_ansi("plain text with no codes"), "plain text with no codes");
        assert_eq!(strip_ansi(""), "");
    }

    #[test]
    fn strip_ansi_handles_truncated_sequences() {
        assert_eq!(strip_ansi("abc\x1b[31"), "abc");
        assert_eq!(strip_ansi("abc\x1b"), "abc");
        assert_eq!(strip_ansi("abc\x1b]0;unterminated"), "abc");
    }
}
