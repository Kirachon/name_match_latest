pub mod connection;
pub mod schema;

#[allow(unused_imports)]
pub use connection::{make_pool, make_pool_with_size};
pub use schema::{
    discover_table_columns, fetch_person_rows_chunk, fetch_person_rows_chunk_all_columns,
    fetch_person_rows_chunk_all_columns_keyset, fetch_person_rows_chunk_keyset,
    fetch_person_rows_chunk_where, fetch_person_rows_chunk_where_keyset, get_max_id,
    get_max_id_where, get_person_count, get_person_rows,
};
