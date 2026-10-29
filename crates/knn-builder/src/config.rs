//! The service's settings, from the environment variables the upstream
//! service reads, under its names and with its defaults, so a deployment's
//! settings carry over; two of knn-builder's own, for Basic Auth; and three
//! for what it knows of the data node: k-NN's client timeout and poll
//! interval, with k-NN's defaults, and the threads it builds on itself.
//!
//! A value the upstream service cannot read stops it at start; so does
//! one here, with the variable named. Numbers are read as Rust reads
//! them, trimmed: Python's underscores between digits are not taken.
//! Where Python takes a value no deployment means, knn-builder refuses
//! it: a TTL or a store size below 0, a memory limit below 0 or not a
//! number. A value that is not UTF-8 is refused too, rather than read as
//! unset, and so is an empty Basic Auth username or password, which would
//! turn on auth anyone could pass.
//!
//! `MAX_WORKERS` and `INDEX_SERIALIZATION_MODE` are read and ignored: the
//! pipeline overlaps one download, one build and one upload instead of
//! running jobs side by side, and streams the index file with no file on
//! disk.

use std::ffi::OsString;
use std::fmt;
use std::time::Duration;

use thiserror::Error;

use crate::cost::CpuThreads;
use crate::service;

/// What the service reads from its environment. The defaults are the
/// upstream service's.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The job store's size and TTL, the memory limits, and the data
    /// node's settings: `REQUEST_STORE_MAX_SIZE`,
    /// `REQUEST_STORE_TTL_SECONDS`, `GPU_MEMORY_LIMIT`,
    /// `CPU_MEMORY_LIMIT`, `CLIENT_TIMEOUT_SECONDS`,
    /// `CLIENT_POLL_INTERVAL_SECONDS` and `CPU_BUILD_THREADS`; their
    /// defaults by default. The GPU is the binary's to read.
    pub service: service::Settings,
    /// `LOG_LEVEL`: `INFO` by default. `DEBUG` adds each job's stage
    /// timings.
    pub log_level: tracing::Level,
    /// `BASIC_AUTH_USERNAME` and `BASIC_AUTH_PASSWORD`, set together:
    /// every request must carry them. Unset by default, and every
    /// request's `Authorization` header is ignored.
    pub auth: Option<Credentials>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            service: service::Settings::default(),
            log_level: tracing::Level::INFO,
            auth: None,
        }
    }
}

/// A Basic Auth username and password.
#[derive(Clone, PartialEq, Eq)]
pub struct Credentials {
    /// The username; no colon, as Basic Auth cannot carry one.
    pub username: String,
    /// The password.
    pub password: String,
}

/// The password stays out of logs.
impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credentials")
            .field("username", &self.username)
            .field("password", &"<hidden>")
            .finish()
    }
}

/// A variable the service cannot start with.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{variable}: {problem}")]
pub struct ConfigError {
    /// The variable's name.
    pub variable: &'static str,
    /// What is wrong with its value.
    pub problem: String,
}

impl Config {
    /// Reads the process's environment.
    ///
    /// # Errors
    ///
    /// The first variable whose value the service cannot start with.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_os(|name| std::env::var_os(name))
    }

    /// Reads the variables `var` gives, as the environment holds them:
    /// a value that is not UTF-8 is refused.
    fn from_os(var: impl Fn(&str) -> Option<OsString>) -> Result<Self, ConfigError> {
        for variable in VARIABLES {
            if var(variable).is_some_and(|value| value.to_str().is_none()) {
                return Err(error(variable, "is not UTF-8"));
            }
        }
        Self::from_vars(|name| var(name).and_then(|value| value.into_string().ok()))
    }

    /// Reads the variables `var` gives, by name.
    ///
    /// # Errors
    ///
    /// The first variable whose value the service cannot start with.
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let mut config = Self::default();
        let settings = &mut config.service;
        if let Some(store_type) = var("REQUEST_STORE_TYPE")
            && store_type != "memory"
        {
            return Err(error("REQUEST_STORE_TYPE", "only `memory` is supported"));
        }
        if let Some(value) = var("REQUEST_STORE_MAX_SIZE") {
            settings.max_jobs = whole("REQUEST_STORE_MAX_SIZE", &value)?;
        }
        if let Some(value) = var("REQUEST_STORE_TTL_SECONDS") {
            settings.ttl = Duration::from_secs(whole("REQUEST_STORE_TTL_SECONDS", &value)?);
        }
        if let Some(value) = var("GPU_MEMORY_LIMIT") {
            settings.gpu_memory_limit = gigabytes("GPU_MEMORY_LIMIT", &value)?;
        }
        if let Some(value) = var("CPU_MEMORY_LIMIT") {
            settings.cpu_memory_limit = gigabytes("CPU_MEMORY_LIMIT", &value)?;
        }
        if let Some(value) = var("CLIENT_TIMEOUT_SECONDS") {
            // With no time at all, every job would be refused.
            let seconds: u64 = whole("CLIENT_TIMEOUT_SECONDS", &value)?;
            if seconds == 0 {
                return Err(error("CLIENT_TIMEOUT_SECONDS", "0 seconds"));
            }
            settings.client_timeout = Duration::from_secs(seconds);
        }
        if let Some(value) = var("CLIENT_POLL_INTERVAL_SECONDS") {
            settings.poll_interval =
                Duration::from_secs(whole("CLIENT_POLL_INTERVAL_SECONDS", &value)?);
        }
        if let Some(value) = var("CPU_BUILD_THREADS") {
            settings.cpu_build = Some(match value.trim() {
                "1" => CpuThreads::One,
                "4" => CpuThreads::Four,
                _ => {
                    return Err(error(
                        "CPU_BUILD_THREADS",
                        &format!(
                            "{value:?} is not 1 or 4, the thread counts the CPU build was measured on"
                        ),
                    ));
                }
            });
        }
        // The service's thread pool refuses 0.
        if let Some(value) = var("MAX_WORKERS")
            && !value
                .trim()
                .parse::<usize>()
                .is_ok_and(|workers| workers >= 1)
        {
            return Err(error(
                "MAX_WORKERS",
                &format!("{value:?} is not a whole number, 1 or more"),
            ));
        }
        if let Some(value) = var("LOG_LEVEL") {
            config.log_level = level(&value)?;
        }
        config.auth = match (var("BASIC_AUTH_USERNAME"), var("BASIC_AUTH_PASSWORD")) {
            (None, None) => None,
            (Some(username), Some(password)) => {
                if username.is_empty() {
                    return Err(error("BASIC_AUTH_USERNAME", "empty"));
                }
                if password.is_empty() {
                    return Err(error("BASIC_AUTH_PASSWORD", "empty"));
                }
                if username.contains(':') {
                    return Err(error(
                        "BASIC_AUTH_USERNAME",
                        "has a colon, which Basic Auth cannot carry",
                    ));
                }
                Some(Credentials { username, password })
            }
            (Some(_), None) => {
                return Err(error("BASIC_AUTH_PASSWORD", "unset, with a username set"));
            }
            (None, Some(_)) => {
                return Err(error("BASIC_AUTH_USERNAME", "unset, with a password set"));
            }
        };
        Ok(config)
    }
}

/// Every variable the service reads: its own, and S3's, which
/// `knn_store::Settings::from_env` reads.
const VARIABLES: [&str; 15] = [
    "REQUEST_STORE_TYPE",
    "REQUEST_STORE_MAX_SIZE",
    "REQUEST_STORE_TTL_SECONDS",
    "GPU_MEMORY_LIMIT",
    "CPU_MEMORY_LIMIT",
    "CLIENT_TIMEOUT_SECONDS",
    "CLIENT_POLL_INTERVAL_SECONDS",
    "CPU_BUILD_THREADS",
    "MAX_WORKERS",
    "INDEX_SERIALIZATION_MODE",
    "LOG_LEVEL",
    "BASIC_AUTH_USERNAME",
    "BASIC_AUTH_PASSWORD",
    "S3_ENDPOINT_URL",
    "AWS_DEFAULT_REGION",
];

fn error(variable: &'static str, problem: &str) -> ConfigError {
    ConfigError {
        variable,
        problem: problem.to_owned(),
    }
}

/// A whole number, 0 or more.
fn whole<T: std::str::FromStr>(variable: &'static str, value: &str) -> Result<T, ConfigError> {
    value.trim().parse().map_err(|_| {
        error(
            variable,
            &format!("{value:?} is not a whole number, 0 or more"),
        )
    })
}

/// GB, as bytes: times 1e9, as the service reads them.
fn gigabytes(variable: &'static str, value: &str) -> Result<f64, ConfigError> {
    match value.trim().parse::<f64>() {
        Ok(gb) if gb.is_finite() && gb >= 0.0 => Ok(gb * 1e9),
        _ => Err(error(
            variable,
            &format!("{value:?} is not a number of GB, 0 or more"),
        )),
    }
}

/// Python's level names, in any case: `CRITICAL` and `FATAL` log errors
/// only, as Rust's logs have nothing above them, and `NOTSET` logs INFO:
/// the service sets it on its own loggers, which then take their root
/// logger's level, INFO.
fn level(value: &str) -> Result<tracing::Level, ConfigError> {
    use tracing::Level;
    match value.trim().to_ascii_uppercase().as_str() {
        "CRITICAL" | "FATAL" | "ERROR" => Ok(Level::ERROR),
        "WARNING" | "WARN" => Ok(Level::WARN),
        "INFO" | "NOTSET" => Ok(Level::INFO),
        "DEBUG" => Ok(Level::DEBUG),
        _ => Err(error(
            "LOG_LEVEL",
            &format!("{value:?} is not one of CRITICAL, ERROR, WARNING, INFO, DEBUG or NOTSET"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::time::Duration;

    use super::{Config, ConfigError, Credentials, VARIABLES};
    use crate::cost::CpuThreads;

    fn read(vars: &[(&str, &str)]) -> Result<Config, ConfigError> {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        Config::from_vars(|name| vars.get(name).cloned())
    }

    #[test]
    fn defaults_to_the_services_settings() {
        assert_eq!(read(&[]), Ok(Config::default()));
    }

    #[test]
    fn reads_each_variable() {
        let config = read(&[
            ("REQUEST_STORE_TYPE", "memory"),
            ("REQUEST_STORE_MAX_SIZE", " 20 "),
            ("REQUEST_STORE_TTL_SECONDS", "60"),
            ("GPU_MEMORY_LIMIT", "8"),
            ("CPU_MEMORY_LIMIT", "64.5"),
            ("CLIENT_TIMEOUT_SECONDS", "600"),
            ("CLIENT_POLL_INTERVAL_SECONDS", "0"),
            ("CPU_BUILD_THREADS", " 4 "),
            ("MAX_WORKERS", "4"),
            ("INDEX_SERIALIZATION_MODE", "MEMORY"),
            ("LOG_LEVEL", "debug"),
            ("BASIC_AUTH_USERNAME", "admin"),
            ("BASIC_AUTH_PASSWORD", "p:w"),
        ])
        .unwrap();
        assert_eq!(config.service.max_jobs, 20);
        assert_eq!(config.service.ttl, Duration::from_secs(60));
        assert_eq!(config.service.gpu_memory_limit, 8e9);
        assert_eq!(config.service.cpu_memory_limit, 64.5e9);
        assert_eq!(config.service.client_timeout, Duration::from_secs(600));
        assert_eq!(config.service.poll_interval, Duration::ZERO);
        assert_eq!(config.service.cpu_build, Some(CpuThreads::Four));
        assert_eq!(config.log_level, tracing::Level::DEBUG);
        assert_eq!(
            config.auth,
            Some(Credentials {
                username: "admin".into(),
                password: "p:w".into(),
            })
        );
    }

    #[test]
    fn names_the_variable_it_cannot_start_with() {
        for (vars, variable) in [
            (&[("REQUEST_STORE_TYPE", "redis")][..], "REQUEST_STORE_TYPE"),
            (
                &[("REQUEST_STORE_MAX_SIZE", "-1")],
                "REQUEST_STORE_MAX_SIZE",
            ),
            (&[("REQUEST_STORE_MAX_SIZE", "")], "REQUEST_STORE_MAX_SIZE"),
            (
                &[("REQUEST_STORE_TTL_SECONDS", "1.5")],
                "REQUEST_STORE_TTL_SECONDS",
            ),
            (&[("GPU_MEMORY_LIMIT", "nan")], "GPU_MEMORY_LIMIT"),
            (&[("CPU_MEMORY_LIMIT", "-1")], "CPU_MEMORY_LIMIT"),
            (&[("CLIENT_TIMEOUT_SECONDS", "0")], "CLIENT_TIMEOUT_SECONDS"),
            (
                &[("CLIENT_TIMEOUT_SECONDS", "1h")],
                "CLIENT_TIMEOUT_SECONDS",
            ),
            (
                &[("CLIENT_POLL_INTERVAL_SECONDS", "-5")],
                "CLIENT_POLL_INTERVAL_SECONDS",
            ),
            (&[("CPU_BUILD_THREADS", "8")], "CPU_BUILD_THREADS"),
            (&[("MAX_WORKERS", "0")], "MAX_WORKERS"),
            (&[("LOG_LEVEL", "LOUD")], "LOG_LEVEL"),
            (&[("BASIC_AUTH_USERNAME", "admin")], "BASIC_AUTH_PASSWORD"),
            (&[("BASIC_AUTH_PASSWORD", "secret")], "BASIC_AUTH_USERNAME"),
            (
                &[
                    ("BASIC_AUTH_USERNAME", ""),
                    ("BASIC_AUTH_PASSWORD", "secret"),
                ],
                "BASIC_AUTH_USERNAME",
            ),
            (
                &[
                    ("BASIC_AUTH_USERNAME", "admin"),
                    ("BASIC_AUTH_PASSWORD", ""),
                ],
                "BASIC_AUTH_PASSWORD",
            ),
            (
                &[("BASIC_AUTH_USERNAME", "a:b"), ("BASIC_AUTH_PASSWORD", "c")],
                "BASIC_AUTH_USERNAME",
            ),
        ] {
            assert_eq!(read(vars).unwrap_err().variable, variable, "{vars:?}");
        }
    }

    #[test]
    fn names_one_floor_for_max_workers() {
        for value in ["0", "-1", "two"] {
            let error = read(&[("MAX_WORKERS", value)]).unwrap_err();
            assert_eq!(
                error.problem,
                format!("{value:?} is not a whole number, 1 or more")
            );
        }
        assert!(read(&[("MAX_WORKERS", " 1 ")]).is_ok());
    }

    #[test]
    fn maps_pythons_level_names() {
        for (name, level) in [
            ("CRITICAL", tracing::Level::ERROR),
            ("ERROR", tracing::Level::ERROR),
            ("WARNING", tracing::Level::WARN),
            ("INFO", tracing::Level::INFO),
            ("NOTSET", tracing::Level::INFO),
        ] {
            assert_eq!(read(&[("LOG_LEVEL", name)]).unwrap().log_level, level);
        }
    }

    /// A value of bytes that are not UTF-8.
    fn not_utf8() -> OsString {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt as _;
            OsString::from_vec(vec![b'a', 0xFF])
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt as _;
            OsString::from_wide(&[u16::from(b'a'), 0xD800])
        }
    }

    /// Every variable read is one whose UTF-8 is checked, and every one
    /// checked, the service's own and S3's aside, is read.
    #[test]
    fn checks_the_variables_it_reads() {
        let asked = std::cell::RefCell::new(Vec::new());
        Config::from_vars(|name| {
            asked.borrow_mut().push(name.to_owned());
            None
        })
        .unwrap();
        let mut asked = asked.into_inner();
        asked.sort();
        let ignored = [
            "INDEX_SERIALIZATION_MODE",
            "S3_ENDPOINT_URL",
            "AWS_DEFAULT_REGION",
        ];
        let mut checked: Vec<String> = VARIABLES
            .iter()
            .filter(|name| !ignored.contains(name))
            .map(|name| (*name).to_owned())
            .collect();
        checked.sort();
        assert_eq!(asked, checked);
    }

    #[test]
    fn refuses_a_value_that_is_not_utf8() {
        for variable in VARIABLES {
            let error = Config::from_os(|name| (name == variable).then(not_utf8)).unwrap_err();
            assert_eq!(error.variable, variable);
        }
        let config = Config::from_os(|name| (name == "LOG_LEVEL").then(|| "DEBUG".into()));
        assert_eq!(config.unwrap().log_level, tracing::Level::DEBUG);
    }

    #[test]
    fn keeps_the_password_out_of_debug_output() {
        let credentials = Credentials {
            username: "admin".into(),
            password: "secret".into(),
        };
        assert!(!format!("{credentials:?}").contains("secret"));
    }
}
