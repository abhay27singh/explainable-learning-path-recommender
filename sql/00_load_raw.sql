-- Load the seven OULAD CSV files into DuckDB tables.
--
-- OULAD encodes missing values as empty strings; read_csv_auto maps those to NULL.
-- Types are declared explicitly rather than sniffed so that a change in the source
-- files surfaces as an error here rather than as silent coercion downstream.

CREATE OR REPLACE TABLE courses AS
SELECT
    code_module::VARCHAR              AS code_module,
    code_presentation::VARCHAR        AS code_presentation,
    module_presentation_length::INT   AS presentation_length
FROM read_csv_auto('${RAW}/courses.csv', header=true);

CREATE OR REPLACE TABLE assessments AS
SELECT
    id_assessment::INT                AS id_assessment,
    code_module::VARCHAR              AS code_module,
    code_presentation::VARCHAR        AS code_presentation,
    assessment_type::VARCHAR          AS assessment_type,
    TRY_CAST(date AS INT)             AS date,          -- NULL for most final exams
    TRY_CAST(weight AS DOUBLE)        AS weight
FROM read_csv_auto('${RAW}/assessments.csv', header=true);

CREATE OR REPLACE TABLE vle AS
SELECT
    id_site::INT                      AS id_site,
    code_module::VARCHAR              AS code_module,
    code_presentation::VARCHAR        AS code_presentation,
    activity_type::VARCHAR            AS activity_type,
    TRY_CAST(week_from AS INT)        AS week_from,     -- present for only ~18% of sites
    TRY_CAST(week_to AS INT)          AS week_to
FROM read_csv_auto('${RAW}/vle.csv', header=true);

CREATE OR REPLACE TABLE student_info AS
SELECT
    id_student::INT                   AS id_student,
    code_module::VARCHAR              AS code_module,
    code_presentation::VARCHAR        AS code_presentation,
    gender::VARCHAR                   AS gender,
    region::VARCHAR                   AS region,
    highest_education::VARCHAR        AS highest_education,
    imd_band::VARCHAR                 AS imd_band,
    age_band::VARCHAR                 AS age_band,
    num_of_prev_attempts::INT         AS num_of_prev_attempts,
    studied_credits::INT              AS studied_credits,
    disability::VARCHAR               AS disability,
    final_result::VARCHAR             AS final_result
FROM read_csv_auto('${RAW}/studentInfo.csv', header=true);

CREATE OR REPLACE TABLE student_registration AS
SELECT
    id_student::INT                   AS id_student,
    code_module::VARCHAR              AS code_module,
    code_presentation::VARCHAR        AS code_presentation,
    TRY_CAST(date_registration AS INT)   AS date_registration,
    TRY_CAST(date_unregistration AS INT) AS date_unregistration
FROM read_csv_auto('${RAW}/studentRegistration.csv', header=true);

CREATE OR REPLACE TABLE student_assessment AS
SELECT
    id_assessment::INT                AS id_assessment,
    id_student::INT                   AS id_student,
    TRY_CAST(date_submitted AS INT)   AS date_submitted,
    is_banked::INT                    AS is_banked,
    TRY_CAST(score AS DOUBLE)         AS score
FROM read_csv_auto('${RAW}/studentAssessment.csv', header=true);

CREATE OR REPLACE TABLE student_vle AS
SELECT
    id_student::INT                   AS id_student,
    code_module::VARCHAR              AS code_module,
    code_presentation::VARCHAR        AS code_presentation,
    id_site::INT                      AS id_site,
    date::INT                         AS date,
    sum_click::INT                    AS sum_click
FROM read_csv_auto('${RAW}/studentVle.csv', header=true);
