-- Derive the concept inventory.
--
-- A concept is a (module, week-block): the set of learning materials a module
-- presents in a given week of study. OULAD ships no concept or knowledge-component
-- annotation, so this is derived rather than read off.
--
-- Week resolution, in priority order:
--   1. vle.week_from, where present. Only ~18% of the 6,364 sites carry it.
--   2. Otherwise the median day on which students actually accessed the site,
--      divided into weeks. This covers the remaining ~82% and reflects observed
--      use rather than course metadata.
--
-- Concepts are keyed on module, not on module-presentation, so that repeated
-- presentations of the same module contribute evidence to the same concept.

CREATE OR REPLACE TABLE site_access_week AS
SELECT
    id_site,
    MEDIAN(date)                    AS median_day,
    COUNT(*)                        AS n_events,
    COUNT(DISTINCT id_student)      AS n_students
FROM student_vle
GROUP BY 1;

CREATE OR REPLACE TABLE site_week AS
SELECT
    v.id_site,
    v.code_module,
    v.activity_type,
    CASE WHEN v.week_from IS NOT NULL THEN 'metadata' ELSE 'observed' END AS week_source,
    GREATEST(0, COALESCE(
        v.week_from,
        CAST(FLOOR(a.median_day / 7.0) AS INT)
    ))                              AS week,
    COALESCE(a.n_events, 0)         AS n_events,
    COALESCE(a.n_students, 0)       AS n_students
FROM vle v
LEFT JOIN site_access_week a USING (id_site)
-- Sites nobody ever accessed and with no metadata week cannot be placed.
WHERE v.week_from IS NOT NULL OR a.median_day IS NOT NULL;

CREATE OR REPLACE TABLE concepts AS
SELECT
    ROW_NUMBER() OVER (ORDER BY code_module, week) - 1  AS concept_id,
    code_module,
    week,
    code_module || '-W' || LPAD(week::VARCHAR, 2, '0')  AS concept_key,
    COUNT(*)                                            AS n_sites,
    SUM(n_events)                                       AS n_events,
    SUM(n_students)                                     AS n_students,
    LIST(DISTINCT activity_type)                        AS activity_types
FROM site_week
GROUP BY code_module, week
HAVING SUM(n_events) > 0
ORDER BY code_module, week;

-- Placement provenance, reported in the paper: how many concepts rest on course
-- metadata versus on observed access behaviour.
CREATE OR REPLACE TABLE concept_provenance AS
SELECT
    week_source,
    COUNT(*)                                            AS n_sites,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)  AS pct
FROM site_week
GROUP BY 1;
