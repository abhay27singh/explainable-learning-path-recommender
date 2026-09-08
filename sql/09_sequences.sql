-- Per-enrolment interaction sequences.
--
-- The unit is an ENROLMENT (student x module x presentation), not a student, because
-- concepts are module-scoped: a student taking two modules has two independent
-- learning trajectories over two disjoint concept sets.
--
-- Full sequences are stored as arrays. Windowing to the model's max length of 200
-- happens in the PyTorch Dataset, so changing that hyperparameter does not require
-- rebuilding the data.

CREATE OR REPLACE TABLE sequences AS
SELECT
    id_student,
    code_module,
    code_presentation,
    COUNT(*)                                              AS n_events,
    SUM((kind = 'assessment')::INT)                       AS n_supervised,
    LIST(concept_id ORDER BY position)                    AS concept_ids,
    LIST(day        ORDER BY position)                    AS days,
    LIST((kind = 'assessment')::TINYINT ORDER BY position) AS is_supervised,
    LIST(COALESCE(label, -1)::TINYINT ORDER BY position)  AS labels,
    LIST(clicks     ORDER BY position)                    AS clicks
FROM events_ordered
GROUP BY 1, 2, 3
-- An enrolment with no supervised token contributes nothing to the loss.
HAVING SUM((kind = 'assessment')::INT) > 0;

CREATE OR REPLACE TABLE sequence_stats AS
SELECT
    COUNT(*)                            AS n_sequences,
    COUNT(DISTINCT id_student)          AS n_students,
    SUM(n_events)                       AS total_events,
    SUM(n_supervised)                   AS total_supervised,
    ROUND(AVG(n_events), 1)             AS mean_events,
    MEDIAN(n_events)                    AS median_events,
    MAX(n_events)                       AS max_events,
    ROUND(AVG(n_supervised), 2)         AS mean_supervised,
    SUM((n_events > 200)::INT)          AS n_over_200
FROM sequences;
