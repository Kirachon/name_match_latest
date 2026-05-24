use name_matcher::export::csv_export::export_to_csv;
use name_matcher::loaders::csv_loader::{CsvEncodingDto, CsvPreviewRequestDto, load_csv_people};
use name_matcher::matching::{MatchOptions, MatchingAlgorithm, match_all_with_opts};

fn write_fixture(name: &str, bytes: &[u8]) -> String {
    let path = std::env::temp_dir().join(format!("nm_csv_e2e_{name}.csv"));
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().to_string()
}

#[test]
fn csv_file_to_column_map_to_run_to_export() {
    let source_path = write_fixture(
        "source",
        b"person_id,given_name,surname,dob,note\n1,Ana,Santos,1990-01-02,source-extra\n",
    );
    let target_path = write_fixture(
        "target",
        b"id,first_name,last_name,birthdate\n10,Ana,Santos,1990-01-02\n",
    );
    let source = load_csv_people(
        &CsvPreviewRequestDto {
            path: source_path,
            encoding: None,
            delimiter: None,
            date_format: None,
        },
        None,
    )
    .unwrap();
    let target = load_csv_people(
        &CsvPreviewRequestDto {
            path: target_path,
            encoding: None,
            delimiter: None,
            date_format: None,
        },
        None,
    )
    .unwrap();

    let pairs = match_all_with_opts(
        &source,
        &target,
        MatchingAlgorithm::FuzzyNoMiddle,
        MatchOptions::default(),
        |_| {},
    );
    assert_eq!(pairs.len(), 1);
    assert_eq!(
        pairs[0]
            .person1
            .extra_fields
            .get("note")
            .map(String::as_str),
        Some("source-extra")
    );

    let export_path = std::env::temp_dir()
        .join("nm_csv_e2e_matches.csv")
        .to_string_lossy()
        .to_string();
    export_to_csv(&pairs, &export_path, MatchingAlgorithm::FuzzyNoMiddle, 0.0).unwrap();
    let exported = std::fs::read_to_string(export_path).unwrap();
    assert!(exported.contains("Ana"));
}

#[test]
fn csv_windows_1252_fixture_runs() {
    let source_path = write_fixture(
        "source_win1252",
        b"id;first_name;last_name;birthdate\n1;Jose\xe9;Reyes;1990-01-02\n",
    );
    let target_path = write_fixture(
        "target_win1252",
        b"id;first_name;last_name;birthdate\n2;Jose\xe9;Reyes;1990-01-02\n",
    );
    let source = load_csv_people(
        &CsvPreviewRequestDto {
            path: source_path,
            encoding: Some(CsvEncodingDto::Windows1252),
            delimiter: None,
            date_format: None,
        },
        None,
    )
    .unwrap();
    let target = load_csv_people(
        &CsvPreviewRequestDto {
            path: target_path,
            encoding: Some(CsvEncodingDto::Windows1252),
            delimiter: None,
            date_format: None,
        },
        None,
    )
    .unwrap();
    let pairs = match_all_with_opts(
        &source,
        &target,
        MatchingAlgorithm::FuzzyNoMiddle,
        MatchOptions::default(),
        |_| {},
    );
    assert_eq!(pairs.len(), 1);
}
