-- EduLens-Next Database Schema
-- Requires PostgreSQL with the pgvector extension for RAG-based retrieval.

-- Enable the pgvector extension for storing and querying embedding vectors
CREATE EXTENSION IF NOT EXISTS pgvector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 1. Student / Learner Profile
--    Stores basic info and the AI-derived personality tags.
-- ============================================================
CREATE TABLE IF NOT EXISTS student_profile (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             VARCHAR(100) NOT NULL,
    birthday         DATE,
    grade            VARCHAR(20),                    -- e.g. "Grade 5", "High School Year 2"
    personality_tags JSONB DEFAULT '{}',             -- AI-inferred traits, e.g. {"patience": 0.8, "focus_style": "visual"}
    learning_style   VARCHAR(50),                    -- "visual", "logical", "auditory"
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 2. Knowledge Graph
--    Represents curriculum knowledge points and their
--    prerequisite / successor relationships.
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_node (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject     VARCHAR(100) NOT NULL,               -- "Math", "English", "Science"
    topic       VARCHAR(255) NOT NULL,               -- "Triangle Angle Sum", "Fractions"
    description TEXT,
    parent_id   UUID REFERENCES knowledge_node(id),  -- prerequisite node
    difficulty  SMALLINT CHECK (difficulty BETWEEN 1 AND 5),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 3. Student Knowledge Map
--    Tracks each student's mastery level for every knowledge node.
--    Status: green (mastered), yellow (shaky), red (gap)
-- ============================================================
CREATE TABLE IF NOT EXISTS student_knowledge_map (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id      UUID NOT NULL REFERENCES student_profile(id) ON DELETE CASCADE,
    knowledge_id    UUID NOT NULL REFERENCES knowledge_node(id),
    mastery_score   NUMERIC(5, 2) DEFAULT 0.0,       -- 0–100 mastery percentage
    status          VARCHAR(10) DEFAULT 'red'         -- 'green', 'yellow', 'red'
                    CHECK (status IN ('green', 'yellow', 'red')),
    last_assessed   TIMESTAMPTZ,
    error_count     INTEGER DEFAULT 0,               -- cumulative wrong answers on this node
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, knowledge_id)
);

-- ============================================================
-- 4. Study Materials
--    Metadata for uploaded homework, exams, textbooks, and notes.
--    The actual file lives in object storage (Minio / Azure Blob).
-- ============================================================
CREATE TABLE IF NOT EXISTS study_material (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id   UUID NOT NULL REFERENCES student_profile(id) ON DELETE CASCADE,
    title        VARCHAR(255),
    file_type    VARCHAR(50) CHECK (file_type IN ('homework', 'exam', 'textbook', 'note', 'other')),
    storage_path TEXT NOT NULL,                      -- path in object storage
    provider     VARCHAR(20) DEFAULT 'minio'         -- 'minio', 'azure', 's3'
                 CHECK (provider IN ('minio', 'azure', 's3')),
    ocr_text     TEXT,                               -- extracted text from OCR
    embedding    VECTOR(1536),                       -- semantic embedding for RAG retrieval
    upload_date  DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 5. Study Sessions
--    Captures each homework / practice session, including timing
--    and AI-generated diagnostics.
-- ============================================================
CREATE TABLE IF NOT EXISTS study_session (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id       UUID NOT NULL REFERENCES student_profile(id) ON DELETE CASCADE,
    material_id      UUID REFERENCES study_material(id),
    start_time       TIMESTAMPTZ NOT NULL,
    end_time         TIMESTAMPTZ,
    duration_seconds INTEGER GENERATED ALWAYS AS (
                         EXTRACT(EPOCH FROM (end_time - start_time))::INTEGER
                     ) STORED,
    accuracy_rate    NUMERIC(5, 2),                  -- percentage of correct answers
    cognitive_load   NUMERIC(3, 2),                  -- AI-assessed load 0.0–1.0
    ai_evaluation    JSONB,                          -- structured diagnosis from LLM
    status           VARCHAR(20) DEFAULT 'in_progress'
                     CHECK (status IN ('in_progress', 'completed', 'struggling', 'timed_out')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 6. Error Items (Wrong Answer Log)
--    Individual wrong answers linked to knowledge nodes and
--    sessions. Drives the spaced-repetition / booster system.
-- ============================================================
CREATE TABLE IF NOT EXISTS error_item (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id       UUID NOT NULL REFERENCES student_profile(id) ON DELETE CASCADE,
    session_id       UUID REFERENCES study_session(id),
    knowledge_id     UUID REFERENCES knowledge_node(id),
    question_text    TEXT NOT NULL,
    student_answer   TEXT,
    correct_answer   TEXT,
    error_category   VARCHAR(50),                    -- 'concept_error', 'calculation', 'logic_gap', 'blind_spot'
    root_cause       TEXT,                           -- AI deep-diagnosis
    remediation      JSONB,                          -- suggested follow-up actions
    resolved         BOOLEAN DEFAULT FALSE,
    embedding        VECTOR(1536),                   -- for similarity search (find similar errors)
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 7. Generated Practice Items (Booster Questions)
--    AI-generated variant questions targeting specific weak spots.
-- ============================================================
CREATE TABLE IF NOT EXISTS practice_item (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id       UUID NOT NULL REFERENCES student_profile(id) ON DELETE CASCADE,
    knowledge_id     UUID REFERENCES knowledge_node(id),
    source_error_id  UUID REFERENCES error_item(id),
    question_text    TEXT NOT NULL,
    expected_answer  TEXT,
    difficulty       SMALLINT CHECK (difficulty BETWEEN 1 AND 5),
    completed        BOOLEAN DEFAULT FALSE,
    student_answer   TEXT,
    is_correct       BOOLEAN,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 8. Daily Study Plans
--    AI-generated daily schedule with prioritised tasks.
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_study_plan (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id    UUID NOT NULL REFERENCES student_profile(id) ON DELETE CASCADE,
    plan_date     DATE NOT NULL,
    plan_content  JSONB NOT NULL,                    -- structured time blocks with subjects and tasks
    generated_by  VARCHAR(50) DEFAULT 'ai',
    completed     BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, plan_date)
);

-- ============================================================
-- 9. Weekly Assessment Reports
--    AI-generated progress snapshots (radar chart data, growth).
-- ============================================================
CREATE TABLE IF NOT EXISTS weekly_report (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id     UUID NOT NULL REFERENCES student_profile(id) ON DELETE CASCADE,
    week_start     DATE NOT NULL,
    radar_data     JSONB NOT NULL,                   -- {comprehension, execution, stability, speed, …}
    summary        TEXT,
    recommendations JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, week_start)
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_session_student ON study_session(student_id);
CREATE INDEX IF NOT EXISTS idx_session_start    ON study_session(start_time);
CREATE INDEX IF NOT EXISTS idx_error_student    ON error_item(student_id);
CREATE INDEX IF NOT EXISTS idx_error_resolved   ON error_item(student_id, resolved);
CREATE INDEX IF NOT EXISTS idx_skm_student      ON student_knowledge_map(student_id);
CREATE INDEX IF NOT EXISTS idx_material_student ON study_material(student_id);

-- Vector similarity index (HNSW – fast approximate nearest-neighbour search)
CREATE INDEX IF NOT EXISTS idx_material_embedding ON study_material USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_error_embedding    ON error_item    USING hnsw (embedding vector_cosine_ops);
