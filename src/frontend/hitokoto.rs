use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Hitokoto {
    pub text: String,
    pub source: Option<String>,
}

#[derive(serde::Deserialize)]
struct HitokotoJson {
    hitokoto: String,
    #[serde(default)]
    from: Option<String>,
    #[serde(default)]
    from_who: Option<String>,
}

fn non_empty(value: &Option<String>) -> Option<String> {
    value
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn build_source(from: &Option<String>, from_who: &Option<String>) -> Option<String> {
    let from = non_empty(from);
    let from_who = non_empty(from_who);
    match (from_who, from) {
        (Some(who), Some(work)) => Some(format!("{who}《{work}》")),
        (Some(who), None) => Some(who),
        (None, Some(work)) => Some(format!("《{work}》")),
        (None, None) => None,
    }
}

pub async fn fetch_hitokoto() -> Option<Hitokoto> {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
    {
        Ok(client) => client,
        Err(_) => return None,
    };
    let json: HitokotoJson = client.get("https://v1.hitokoto.cn/").send().await.ok()?.json().await.ok()?;
    let text = json.hitokoto.trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some(Hitokoto {
        text,
        source: build_source(&json.from, &json.from_who),
    })
}

pub const FALLBACKS: &[(&str, Option<&str>)] = &[
    ("学而不思则罔，思而不学则殆。", Some("孔子《论语》")),
    ("路漫漫其修远兮，吾将上下而求索。", Some("屈原《离骚》")),
    ("人生到处知何似，应似飞鸿踏雪泥。", Some("苏轼《和子由渑池怀旧》")),
    ("竹杖芒鞋轻胜马，谁怕？一蓑烟雨任平生。", Some("苏轼《定风波》")),
    ("为天地立心，为生民立命，为往圣继绝学，为万世开太平。", Some("张载")),
    ("黑夜给了我黑色的眼睛，我却用它寻找光明。", Some("顾城《一代人》")),
    ("面朝大海，春暖花开。", Some("海子")),
    ("世界上只有一种真正的英雄主义，那就是认清生活的真相后依然热爱生活。", Some("罗曼·罗兰")),
];

pub fn fallback_hitokoto() -> Hitokoto {
    let (text, source) = FALLBACKS[fastrand::usize(..FALLBACKS.len())];
    Hitokoto {
        text: text.to_string(),
        source: source.map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallbacks_are_non_empty_and_unique() {
        assert!(!FALLBACKS.is_empty());
        let mut seen = std::collections::HashSet::new();
        for (text, _) in FALLBACKS {
            assert!(!text.is_empty());
            assert!(seen.insert(*text), "duplicate fallback: {text}");
        }
    }

    #[test]
    fn source_formats_combinations() {
        assert_eq!(
            build_source(&Some("临安春雨初霁".into()), &Some("陆游".into())),
            Some("陆游《临安春雨初霁》".to_string())
        );
        assert_eq!(
            build_source(&Some("临安春雨初霁".into()), &None),
            Some("《临安春雨初霁》".to_string())
        );
        assert_eq!(
            build_source(&None, &Some("陆游".into())),
            Some("陆游".to_string())
        );
        assert_eq!(build_source(&None, &None), None);
        assert_eq!(build_source(&Some("  ".into()), &Some("".into())), None);
    }
}
