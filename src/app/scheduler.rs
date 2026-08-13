use std::time::{Duration, Instant};

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    OutputContains { cmd: String, contains: String },
    ExitZero { cmd: String },
    FileExists { path: String },
}

impl Condition {
    pub fn holds(&self) -> bool {
        match self {
            Self::OutputContains { cmd, contains } => run(cmd)
                .map(|out| out.contains(contains))
                .unwrap_or(false),
            Self::ExitZero { cmd } => run(cmd).is_some(),
            Self::FileExists { path } => std::path::Path::new(path).exists(),
        }
    }
}

fn run(cmd: &str) -> Option<String> {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub enum TaskKind {
    Delay { due: Instant },
    Schedule { interval: Duration, next: Instant },
    Monitor { condition: Condition, check_every: Duration, deadline: Option<Instant>, last_check: Option<Instant>, checking: bool },
}

#[derive(Debug, Clone)]
pub struct TaskAction {
    pub tool: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub id: u64,
    pub kind: TaskKind,
    pub action: TaskAction,
}

impl ScheduledTask {
    pub fn describe(&self) -> String {
        let kind = match &self.kind {
            TaskKind::Delay { .. } => "delay".to_string(),
            TaskKind::Schedule { interval, .. } => format!("every {}s", interval.as_secs()),
            TaskKind::Monitor { condition, .. } => {
                let cond = match condition {
                    Condition::OutputContains { cmd, contains } => {
                        format!("output of \"{cmd}\" contains \"{contains}\"")
                    }
                    Condition::ExitZero { cmd } => format!("\"{cmd}\" exits 0"),
                    Condition::FileExists { path } => format!("\"{path}\" exists"),
                };
                format!("monitor: {cond}")
            }
        };
        format!("#{} {} -> {}({})", self.id, kind, self.action.tool, self.action.arguments)
    }
}

pub enum Fired {
    Execute { id: u64, action: TaskAction, label: String },
    MonitorTimeout { id: u64 },
    Check { id: u64, condition: Condition },
}

pub struct Scheduler {
    tasks: Vec<ScheduledTask>,
    next_id: u64,
}

enum Op {
    None,
    Remove { id: u64, action: TaskAction, label: String },
    FireKeep { id: u64, action: TaskAction, label: String },
    Timeout { id: u64 },
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            next_id: 1,
        }
    }

    pub fn register(&mut self, kind: TaskKind, action: TaskAction) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.tasks.push(ScheduledTask { id, kind, action });
        id
    }

    pub fn cancel(&mut self, id: u64) -> bool {
        let before = self.tasks.len();
        self.tasks.retain(|t| t.id != id);
        self.tasks.len() != before
    }

    pub fn tasks(&self) -> &[ScheduledTask] {
        &self.tasks
    }

    pub fn poll(&mut self, now: Instant) -> Vec<Fired> {
        let mut fired = Vec::new();
        let mut i = 0;
        while i < self.tasks.len() {
            let mut op = Op::None;
            let mut reschedule: Option<Instant> = None;
            match &self.tasks[i].kind {
                TaskKind::Delay { due } => {
                    if now >= *due {
                        op = Op::Remove {
                            id: self.tasks[i].id,
                            action: self.tasks[i].action.clone(),
                            label: "delay".into(),
                        };
                    }
                }
                TaskKind::Schedule { interval, next } => {
                    if now >= *next {
                        op = Op::FireKeep {
                            id: self.tasks[i].id,
                            action: self.tasks[i].action.clone(),
                            label: "schedule tick".into(),
                        };
                        reschedule = Some(now + *interval);
                    }
                }
                TaskKind::Monitor { .. } => {}
            }
            let task_id = self.tasks[i].id;
            if let TaskKind::Monitor {
                check_every,
                deadline,
                last_check,
                checking,
                condition,
            } = &mut self.tasks[i].kind
            {
                if deadline.is_some_and(|d| now >= d) {
                    op = Op::Timeout { id: task_id };
                } else if !*checking {
                    let due = match last_check {
                        Some(last) => now >= *last + *check_every,
                        None => true,
                    };
                    if due {
                        *checking = true;
                        fired.push(Fired::Check {
                            id: task_id,
                            condition: condition.clone(),
                        });
                    }
                }
            }
            match op {
                Op::Remove { id, action, label } => {
                    self.tasks.remove(i);
                    fired.push(Fired::Execute { id, action, label });
                    continue;
                }
                Op::FireKeep { id, action, label } => {
                    if let Some(next) = reschedule
                        && let TaskKind::Schedule { next: slot, .. } = &mut self.tasks[i].kind {
                            *slot = next;
                        }
                    fired.push(Fired::Execute { id, action, label });
                }
                Op::Timeout { id } => {
                    self.tasks.remove(i);
                    fired.push(Fired::MonitorTimeout { id });
                    continue;
                }
                Op::None => {}
            }
            i += 1;
        }
        fired
    }

    pub fn check_result(&mut self, id: u64, holds: bool, now: Instant) -> Vec<Fired> {
        let mut fired = Vec::new();
        if let Some(task) = self.tasks.iter_mut().find(|task| task.id == id)
            && let TaskKind::Monitor { last_check, checking, .. } = &mut task.kind
        {
            *checking = false;
            if holds {
                let action = task.action.clone();
                let label = "monitor matched".to_string();
                let tid = task.id;
                self.tasks.retain(|t| t.id != tid);
                fired.push(Fired::Execute { id: tid, action, label });
            } else {
                *last_check = Some(now);
            }
        }
        fired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action() -> TaskAction {
        TaskAction {
            tool: "time".into(),
            arguments: Value::Object(Default::default()),
        }
    }

    #[test]
    fn delay_fires_once_then_removed() {
        let mut scheduler = Scheduler::new();
        scheduler.register(
            TaskKind::Delay { due: Instant::now() + Duration::from_secs(30) },
            action(),
        );
        assert!(scheduler.poll(Instant::now()).is_empty());
        let fired = scheduler.poll(Instant::now() + Duration::from_secs(31));
        assert_eq!(fired.len(), 1);
        assert!(scheduler.tasks().is_empty());
    }

    #[test]
    fn schedule_fires_repeatedly_until_cancelled() {
        let mut scheduler = Scheduler::new();
        let id = scheduler.register(
            TaskKind::Schedule {
                interval: Duration::from_secs(10),
                next: Instant::now() + Duration::from_secs(10),
            },
            action(),
        );
        let now = Instant::now();
        assert!(scheduler.poll(now).is_empty());
        let fired = scheduler.poll(now + Duration::from_secs(11));
        assert_eq!(fired.len(), 1);
        assert!(scheduler.cancel(id));
        assert!(!scheduler.cancel(id));
        assert!(scheduler.poll(now + Duration::from_secs(100)).is_empty());
    }

    #[test]
    fn monitor_output_contains_fires_when_condition_holds() {
        let mut scheduler = Scheduler::new();
        scheduler.register(
            TaskKind::Monitor {
                condition: Condition::OutputContains {
                    cmd: "echo hello world".into(),
                    contains: "hello".into(),
                },
                check_every: Duration::from_secs(1),
                deadline: Some(Instant::now() + Duration::from_secs(60)),
                last_check: None,
                checking: false,
            },
            action(),
        );
        let now = Instant::now();
        let fired = scheduler.poll(now);
        assert!(matches!(fired[0], Fired::Check { .. }), "poll must request an async check");
        let id = match &fired[0] {
            Fired::Check { id, .. } => *id,
            _ => unreachable!(),
        };
        let matched = scheduler.check_result(id, true, now);
        assert_eq!(matched.len(), 1);
        assert!(matches!(matched[0], Fired::Execute { .. }));
        assert!(scheduler.tasks().is_empty());
    }

    #[test]
    fn monitor_waits_until_condition_holds() {
        let mut scheduler = Scheduler::new();
        let path = std::env::temp_dir().join(format!("scheduler_test_{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        scheduler.register(
            TaskKind::Monitor {
                condition: Condition::FileExists { path: path.display().to_string() },
                check_every: Duration::from_secs(1),
                deadline: Some(Instant::now() + Duration::from_secs(60)),
                last_check: None,
                checking: false,
            },
            action(),
        );
        let now = Instant::now();
        let fired = scheduler.poll(now);
        assert_eq!(fired.len(), 1, "first poll requests a check");
        let id = match &fired[0] {
            Fired::Check { id, .. } => *id,
            _ => unreachable!(),
        };
        assert!(scheduler.check_result(id, false, now).is_empty());
        assert_eq!(scheduler.tasks().len(), 1, "missed check keeps the task");
        std::fs::write(&path, "x").unwrap();
        let fired = scheduler.poll(now + Duration::from_secs(2));
        assert_eq!(fired.len(), 1, "due again after check_every");
        let id = match &fired[0] {
            Fired::Check { id, .. } => *id,
            _ => unreachable!(),
        };
        assert_eq!(scheduler.check_result(id, true, now).len(), 1);
        assert!(scheduler.tasks().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn monitor_times_out_and_is_removed() {
        let mut scheduler = Scheduler::new();
        scheduler.register(
            TaskKind::Monitor {
                condition: Condition::ExitZero { cmd: "false".into() },
                check_every: Duration::from_secs(1),
                deadline: Some(Instant::now() + Duration::from_secs(5)),
                last_check: None,
                checking: false,
            },
            action(),
        );
        let now = Instant::now();
        assert_eq!(scheduler.poll(now).len(), 1, "first poll requests a check");
        let fired = scheduler.poll(now + Duration::from_secs(6));
        assert!(matches!(fired[0], Fired::MonitorTimeout { .. }));
        assert!(scheduler.tasks().is_empty());
    }

    #[test]
    fn monitor_rechecks_only_after_check_every() {
        let mut scheduler = Scheduler::new();
        let path = std::env::temp_dir().join(format!("scheduler_recheck_{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        scheduler.register(
            TaskKind::Monitor {
                condition: Condition::FileExists { path: path.display().to_string() },
                check_every: Duration::from_secs(10),
                deadline: Some(Instant::now() + Duration::from_secs(60)),
                last_check: None,
                checking: false,
            },
            action(),
        );
        let now = Instant::now();
        let fired = scheduler.poll(now);
        assert_eq!(fired.len(), 1, "first poll requests a check");
        let id = match &fired[0] {
            Fired::Check { id, .. } => *id,
            _ => unreachable!(),
        };
        assert!(scheduler.check_result(id, false, now).is_empty());
        std::fs::write(&path, "x").unwrap();
        assert!(scheduler.poll(now + Duration::from_secs(2)).is_empty(), "no check before check_every");
        let fired = scheduler.poll(now + Duration::from_secs(12));
        assert_eq!(fired.len(), 1, "check requested again after check_every");
        let id = match &fired[0] {
            Fired::Check { id, .. } => *id,
            _ => unreachable!(),
        };
        assert_eq!(scheduler.check_result(id, true, now).len(), 1);
        let _ = std::fs::remove_file(&path);
    }
}
