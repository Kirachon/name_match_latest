use name_matcher::export::csv_export::export_to_csv;
use name_matcher::loaders::excel_loader::{ExcelPreviewRequestDto, load_excel_people};
use name_matcher::matching::{MatchOptions, MatchingAlgorithm, match_all_with_opts};
use rust_xlsxwriter::{Formula, Workbook};

fn write_source_fixture(name: &str) -> String {
    let path = std::env::temp_dir().join(format!("nm_excel_e2e_{name}_source.xlsx"));
    let mut workbook = Workbook::new();
    {
        let ignored = workbook.add_worksheet();
        ignored.set_name("Ignored").unwrap();
        ignored.write_string(0, 0, "not_used").unwrap();
        ignored.write_string(1, 0, "skip").unwrap();
    }
    {
        let people = workbook.add_worksheet();
        people.set_name("People").unwrap();
        people.write_string(0, 0, "person_id").unwrap();
        people.write_string(0, 1, "given_name").unwrap();
        people.write_string(0, 2, "surname").unwrap();
        people.write_string(0, 3, "dob").unwrap();
        people.write_string(0, 4, "note").unwrap();
        people.write_number(1, 0, 1).unwrap();
        people
            .write_formula(1, 1, Formula::new("\"Ana\"").set_result("Ana"))
            .unwrap();
        people.write_string(1, 2, "Santos").unwrap();
        people.write_number(1, 3, 32875).unwrap();
        people.write_string(1, 4, "source-extra").unwrap();
    }
    workbook.save(&path).unwrap();
    path.to_string_lossy().to_string()
}

fn write_target_fixture(name: &str) -> String {
    let path = std::env::temp_dir().join(format!("nm_excel_e2e_{name}_target.xlsx"));
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    sheet.set_name("People").unwrap();
    sheet.write_string(0, 0, "id").unwrap();
    sheet.write_string(0, 1, "first_name").unwrap();
    sheet.write_string(0, 2, "last_name").unwrap();
    sheet.write_string(0, 3, "birthdate").unwrap();
    sheet.write_number(1, 0, 10).unwrap();
    sheet.write_string(1, 1, "Ana").unwrap();
    sheet.write_string(1, 2, "Santos").unwrap();
    sheet.write_string(1, 3, "1990-01-02").unwrap();
    workbook.save(&path).unwrap();
    path.to_string_lossy().to_string()
}

#[test]
fn excel_file_to_column_map_to_run_to_export() {
    let source_path = write_source_fixture("run");
    let target_path = write_target_fixture("run");
    let source = load_excel_people(
        &ExcelPreviewRequestDto {
            path: source_path,
            sheet_name: Some("People".to_string()),
            date_format: None,
        },
        None,
    )
    .unwrap();
    let target = load_excel_people(
        &ExcelPreviewRequestDto {
            path: target_path,
            sheet_name: Some("People".to_string()),
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
        .join("nm_excel_e2e_matches.csv")
        .to_string_lossy()
        .to_string();
    export_to_csv(&pairs, &export_path, MatchingAlgorithm::FuzzyNoMiddle, 0.0).unwrap();
    let exported = std::fs::read_to_string(export_path).unwrap();
    assert!(exported.contains("Ana"));
}
