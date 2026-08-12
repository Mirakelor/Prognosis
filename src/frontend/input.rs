use crate::frontend::state::InputState;

pub fn insert_char(state: &mut InputState, c: char) {
    state.buffer.insert(char_index_to_byte(&state.buffer, state.cursor), c);
    state.cursor += 1;
}

pub fn insert_text(state: &mut InputState, text: &str) {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let byte = char_index_to_byte(&state.buffer, state.cursor);
    state.buffer.insert_str(byte, &normalized);
    state.cursor += normalized.chars().count();
}

pub fn backspace(state: &mut InputState) {
    if state.cursor == 0 {
        return;
    }
    let byte = char_index_to_byte(&state.buffer, state.cursor);
    let prev = state.buffer[..byte]
        .chars()
        .next_back()
        .map(|c| c.len_utf8())
        .unwrap_or(0);
    state.buffer.replace_range(byte - prev..byte, "");
    state.cursor -= 1;
}

pub fn delete_char(state: &mut InputState) {
    let byte = char_index_to_byte(&state.buffer, state.cursor);
    if byte >= state.buffer.len() {
        return;
    }
    let next = state.buffer[byte..].chars().next().map(|c| c.len_utf8()).unwrap_or(0);
    state.buffer.replace_range(byte..byte + next, "");
}

pub fn move_left(state: &mut InputState) {
    if state.cursor > 0 {
        state.cursor -= 1;
    }
}

pub fn move_right(state: &mut InputState) {
    if state.cursor < state.buffer.chars().count() {
        state.cursor += 1;
    }
}

pub fn move_home(state: &mut InputState) {
    state.cursor = 0;
}

pub fn move_end(state: &mut InputState) {
    state.cursor = state.buffer.chars().count();
}

pub fn newline(state: &mut InputState) {
    insert_char(state, '\n');
}

pub fn history_previous(state: &mut InputState) {
    if state.history.is_empty() {
        return;
    }
    let index = match state.history_index {
        Some(index) if index > 0 => index - 1,
        _ => state.history.len() - 1,
    };
    state.history_index = Some(index);
    state.buffer = state.history[index].clone();
    state.cursor = state.buffer.chars().count();
}

pub fn history_next(state: &mut InputState) {
    let Some(index) = state.history_index else {
        return;
    };
    if index + 1 < state.history.len() {
        state.history_index = Some(index + 1);
        state.buffer = state.history[index + 1].clone();
    } else {
        state.history_index = None;
        state.buffer.clear();
    }
    state.cursor = state.buffer.chars().count();
}

pub fn commit(state: &mut InputState) -> Option<String> {
    let text = state.buffer.trim().to_string();
    if text.is_empty() {
        return None;
    }
    if !text.starts_with('/')
        && state.history.last() != Some(&text) {
            state.history.push(text.clone());
        }
    state.buffer.clear();
    state.cursor = 0;
    state.history_index = None;
    Some(text)
}

fn char_index_to_byte(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_and_editing() {
        let mut state = InputState::new();
        for c in "hello".chars() {
            insert_char(&mut state, c);
        }
        assert_eq!(state.buffer, "hello");
        assert_eq!(state.cursor, 5);
        move_left(&mut state);
        assert_eq!(state.cursor, 4);
        backspace(&mut state);
        assert_eq!(state.buffer, "helo");
        insert_char(&mut state, 'i');
        assert_eq!(state.buffer, "helio");
        move_home(&mut state);
        insert_char(&mut state, 'H');
        assert_eq!(state.buffer, "Hhelio");
        move_end(&mut state);
        delete_char(&mut state);
        assert_eq!(state.buffer, "Hhelio");
        backspace(&mut state);
        assert_eq!(state.buffer, "Hheli");
    }

    #[test]
    fn multi_line_insertion() {
        let mut state = InputState::new();
        insert_text(&mut state, "line1");
        newline(&mut state);
        insert_text(&mut state, "line2");
        assert_eq!(state.buffer, "line1\nline2");
        assert_eq!(state.cursor, 11);
        move_home(&mut state);
        newline(&mut state);
        assert_eq!(state.buffer, "\nline1\nline2");
    }

    #[test]
    fn paste_carriage_returns_normalized_to_newlines() {
        let mut state = InputState::new();
        insert_text(&mut state, "first\r\nsecond\rthird");
        assert_eq!(state.buffer, "first\nsecond\nthird");
        assert_eq!(state.cursor, 18);
    }

    #[test]
    fn history_navigation() {
        let mut state = InputState::new();
        assert!(commit(&mut state).is_none());
        insert_text(&mut state, "first");
        assert_eq!(commit(&mut state).as_deref(), Some("first"));
        insert_text(&mut state, "second");
        assert_eq!(commit(&mut state).as_deref(), Some("second"));
        assert_eq!(state.history.len(), 2);
        history_previous(&mut state);
        assert_eq!(state.buffer, "second");
        history_previous(&mut state);
        assert_eq!(state.buffer, "first");
        history_next(&mut state);
        assert_eq!(state.buffer, "second");
        history_next(&mut state);
        assert_eq!(state.buffer, "");
    }

    #[test]
    fn command_text_is_not_saved_to_history() {
        let mut state = InputState::new();
        insert_text(&mut state, "/status");
        assert_eq!(commit(&mut state).as_deref(), Some("/status"));
        assert!(state.history.is_empty());
    }
}
