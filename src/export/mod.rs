use crate::matching::{MatchPair, MatchingAlgorithm};
use std::collections::BTreeSet;

pub mod csv_export;
pub mod xlsx_export;

pub(crate) fn collect_extra_field_names(results: &[MatchPair]) -> Vec<String> {
    let mut field_set = BTreeSet::new();
    for pair in results {
        for key in pair.person2.extra_fields.keys() {
            field_set.insert(key.clone());
        }
    }
    field_set.into_iter().collect()
}

pub(crate) fn build_match_headers(
    algorithm: MatchingAlgorithm,
    extra_field_names: &[String],
) -> Vec<String> {
    let mut headers: Vec<String> = match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => vec![
            "Table1_ID".to_string(),
            "Table1_UUID".to_string(),
            "Table1_FirstName".to_string(),
            "Table1_LastName".to_string(),
            "Table1_Birthdate".to_string(),
            "Table2_ID".to_string(),
            "Table2_UUID".to_string(),
            "Table2_FirstName".to_string(),
            "Table2_LastName".to_string(),
            "Table2_Birthdate".to_string(),
        ],
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => vec![
            "Table1_ID".to_string(),
            "Table1_UUID".to_string(),
            "Table1_FirstName".to_string(),
            "Table1_MiddleName".to_string(),
            "Table1_LastName".to_string(),
            "Table1_Birthdate".to_string(),
            "Table2_ID".to_string(),
            "Table2_UUID".to_string(),
            "Table2_FirstName".to_string(),
            "Table2_MiddleName".to_string(),
            "Table2_LastName".to_string(),
            "Table2_Birthdate".to_string(),
        ],
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::HouseholdGpu
        | MatchingAlgorithm::HouseholdGpuOpt6
        | MatchingAlgorithm::LevenshteinWeighted => vec![
            "Table1_ID".to_string(),
            "Table1_UUID".to_string(),
            "Table1_FirstName".to_string(),
            "Table1_MiddleName".to_string(),
            "Table1_LastName".to_string(),
            "Table1_Birthdate".to_string(),
            "Table2_ID".to_string(),
            "Table2_UUID".to_string(),
            "Table2_FirstName".to_string(),
            "Table2_MiddleName".to_string(),
            "Table2_LastName".to_string(),
            "Table2_Birthdate".to_string(),
        ],
    };

    for field_name in extra_field_names {
        headers.push(format!("Table2_{}", field_name));
    }

    match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            headers.push("is_matched_Infnbd".to_string());
            headers.push("Confidence".to_string());
            headers.push("MatchedFields".to_string());
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            headers.push("is_matched_Infnmnbd".to_string());
            headers.push("Confidence".to_string());
            headers.push("MatchedFields".to_string());
        }
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::LevenshteinWeighted
        | MatchingAlgorithm::HouseholdGpu
        | MatchingAlgorithm::HouseholdGpuOpt6 => {
            headers.push("is_matched_Fuzzy".to_string());
            headers.push("Confidence".to_string());
            headers.push("MatchedFields".to_string());
        }
    }

    headers
}

#[cfg(test)]
mod tests {
    use super::{build_match_headers, collect_extra_field_names};
    use crate::export::csv_export::export_to_csv;
    use crate::export::xlsx_export::{export_to_xlsx, SummaryContext};
    use crate::matching::{MatchPair, MatchingAlgorithm};
    use crate::models::Person;
    use chrono::{NaiveDate, Utc};
    use std::collections::HashMap;

    fn person(
        id: i64,
        first: &str,
        middle: Option<&str>,
        last: &str,
        birthdate: (i32, u32, u32),
        extra_fields: HashMap<String, String>,
    ) -> Person {
        Person {
            id,
            uuid: Some(format!("u{id}")),
            first_name: Some(first.to_string()),
            middle_name: middle.map(|s| s.to_string()),
            last_name: Some(last.to_string()),
            birthdate: NaiveDate::from_ymd_opt(birthdate.0, birthdate.1, birthdate.2),
            hh_id: None,
            extra_fields,
        }
    }

    #[test]
    fn export_csv_xlsx_parity_headers() {
        let mut extra = HashMap::new();
        extra.insert("region_code".to_string(), "R1".to_string());
        extra.insert("poor_hat_0".to_string(), "1".to_string());
        extra.insert("custom_field".to_string(), "X".to_string());

        let p1 = person(1, "Alice", None, "Smith", (1990, 1, 1), HashMap::new());
        let p2 = person(2, "Alice", None, "Smith", (1990, 1, 1), extra);

        let matches = vec![MatchPair {
            person1: p1,
            person2: p2,
            confidence: 100.0,
            matched_fields: vec![
                "id".to_string(),
                "uuid".to_string(),
                "first_name".to_string(),
                "last_name".to_string(),
                "birthdate".to_string(),
            ],
            is_matched_infnbd: true,
            is_matched_infnmnbd: false,
        }];

        let csv_path = "./target/test_export_parity.csv";
        let xlsx_path = "./target/test_export_parity.xlsx";
        let _ = std::fs::remove_file(csv_path);
        let _ = std::fs::remove_file(xlsx_path);

        export_to_csv(
            &matches,
            csv_path,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
            0.0,
        )
        .expect("csv export failed");

        let summary = SummaryContext {
            db_name: "db".into(),
            table1: "t1".into(),
            table2: "t2".into(),
            total_table1: 1,
            total_table2: 1,
            matches_algo1: 1,
            matches_algo2: 0,
            matches_fuzzy: 0,
            overlap_count: 0,
            unique_algo1: 1,
            unique_algo2: 0,
            fetch_time: std::time::Duration::from_millis(1),
            match1_time: std::time::Duration::from_millis(1),
            match2_time: std::time::Duration::from_millis(1),
            export_time: std::time::Duration::from_millis(1),
            mem_used_start_mb: 0,
            mem_used_end_mb: 0,
            started_utc: Utc::now(),
            ended_utc: Utc::now(),
            duration_secs: 0.0,
            exec_mode_algo1: Some("CPU".into()),
            exec_mode_algo2: Some("CPU".into()),
            exec_mode_fuzzy: None,
            algo_used: "Algo1".into(),
            gpu_used: false,
            gpu_total_mb: 0,
            gpu_free_mb_end: 0,
            adv_level: None,
            adv_level_description: None,
        };

        export_to_xlsx(&matches, &[], xlsx_path, &summary).expect("xlsx export failed");

        let mut rdr = csv::Reader::from_path(csv_path).expect("read csv");
        let headers: Vec<String> = rdr
            .headers()
            .expect("csv headers")
            .iter()
            .map(|s| s.to_string())
            .collect();
        let extra_fields = collect_extra_field_names(&matches);
        let expected =
            build_match_headers(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &extra_fields);
        assert_eq!(headers, expected, "csv headers should match shared header builder");

        let csv_meta = std::fs::metadata(csv_path).expect("csv metadata");
        assert!(csv_meta.len() > 0);
        let xlsx_meta = std::fs::metadata(xlsx_path).expect("xlsx metadata");
        assert!(xlsx_meta.len() > 0);
    }
}
