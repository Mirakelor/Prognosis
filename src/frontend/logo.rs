pub const TIPS: &[&str] = &[
    "Type / to see available commands",
    "Shift+Enter / Alt+Enter inserts a newline in the input box",
    "Esc interrupts the current generation",
    "Some tool calls ask for approval: Enter approves, Shift+Enter remembers, Esc denies",
    "The /status panel shows live cognitive signals: RPE, modulators, memory",
    "Conversation history is always kept in full; /compact is the only way to compress it",
    "Rules (project + ~/.prognosis) are injected automatically when they apply",
    "Use /models to switch models without restarting",
    "Scheduled tasks keep running and report into the conversation",
    "Use /history to reload a past session, /continue for the most recent one",
    "Use /remember to inject a summarized memory from an archived session",
    "Paste inserts text without sending; press Enter to send",
];

pub fn pick_tip() -> &'static str {
    let index = fastrand::usize(..TIPS.len());
    TIPS[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tips_are_non_empty_and_unique() {
        assert!(!TIPS.is_empty());
        let mut seen = std::collections::HashSet::new();
        for tip in TIPS {
            assert!(!tip.is_empty());
            assert!(seen.insert(*tip), "duplicate tip: {tip}");
        }
    }

    #[test]
    fn picked_tip_is_from_list() {
        let tip = pick_tip();
        assert!(TIPS.contains(&tip));
    }
}
