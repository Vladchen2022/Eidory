PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_path TEXT NOT NULL UNIQUE,
    import_mode TEXT NOT NULL DEFAULT 'indexed',
    added_at TEXT NOT NULL,
    last_scanned_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    CHECK(import_mode IN ('indexed', 'copied', 'moved'))
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id INTEGER NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_ext TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    duration_ms INTEGER,
    created_at TEXT,
    modified_at TEXT,
    modified_time_ns INTEGER NOT NULL,
    imported_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    thumbnail_path TEXT,
    thumbnail_status TEXT NOT NULL DEFAULT 'pending',
    embedding_status TEXT NOT NULL DEFAULT 'pending',
    is_missing INTEGER NOT NULL DEFAULT 0,
    is_favorite INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    FOREIGN KEY(folder_id) REFERENCES folders(id) ON DELETE CASCADE,
    CHECK(thumbnail_status IN ('pending', 'ready', 'failed')),
    CHECK(embedding_status IN ('pending', 'processing', 'ready', 'failed', 'stale'))
);

CREATE INDEX IF NOT EXISTS idx_images_folder_id ON images(folder_id);
CREATE INDEX IF NOT EXISTS idx_images_missing ON images(is_missing);
CREATE INDEX IF NOT EXISTS idx_images_favorite ON images(is_favorite);
CREATE INDEX IF NOT EXISTS idx_images_embedding_status ON images(embedding_status);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    vector_blob BLOB,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(image_id, model_name, model_revision, embedding_dim),
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    CHECK(status IN ('pending', 'processing', 'ready', 'failed', 'stale'))
);

CREATE INDEX IF NOT EXISTS idx_embeddings_lookup
ON embeddings(model_name, model_revision, embedding_dim, status);

CREATE TABLE IF NOT EXISTS color_features (
    image_id INTEGER PRIMARY KEY,
    vector_version TEXT NOT NULL,
    vector_dim INTEGER NOT NULL,
    hist_blob BLOB,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    CHECK(status IN ('pending', 'ready', 'failed', 'stale'))
);

CREATE INDEX IF NOT EXISTS idx_color_features_lookup
ON color_features(vector_version, vector_dim, status);

CREATE TABLE IF NOT EXISTS ai_vision_collection_rules (
    collection_id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL,
    include_descendants INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    CHECK(mode IN ('include', 'exclude'))
);

CREATE INDEX IF NOT EXISTS idx_ai_vision_collection_rules_mode
ON ai_vision_collection_rules(mode);

CREATE TABLE IF NOT EXISTS ai_vision_tags (
    image_id INTEGER PRIMARY KEY,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    scene_location TEXT,
    environment_type TEXT,
    time_of_day TEXT,
    weather TEXT,
    shot_scale TEXT,
    view_angle TEXT,
    lighting_json TEXT NOT NULL DEFAULT '[]',
    confidence_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT,
    error_message TEXT,
    source_modified_time_ns INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    CHECK(status IN ('pending', 'processing', 'ready', 'failed', 'stale', 'skipped'))
);

CREATE INDEX IF NOT EXISTS idx_ai_vision_tags_status
ON ai_vision_tags(status);

CREATE INDEX IF NOT EXISTS idx_ai_vision_tags_fields
ON ai_vision_tags(scene_location, environment_type, time_of_day, weather, shot_scale, view_angle);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE,
    tag_type TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    CHECK(tag_type IN ('user', 'project', 'style', 'usage'))
);

CREATE TABLE IF NOT EXISTS image_tags (
    image_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    confirmed_by_user INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    PRIMARY KEY(image_id, tag_id),
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    CHECK(source IN ('manual', 'ai_confirmed'))
);

CREATE INDEX IF NOT EXISTS idx_image_tags_tag_id ON image_tags(tag_id);

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,
    name TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(parent_id) REFERENCES collections(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_collections_parent_order
ON collections(parent_id, sort_order, name COLLATE NOCASE);

CREATE UNIQUE INDEX IF NOT EXISTS idx_collections_parent_name_unique
ON collections(COALESCE(parent_id, 0), name COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS image_collections (
    image_id INTEGER NOT NULL,
    collection_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(image_id, collection_id),
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_collections_collection_id
ON image_collections(collection_id);

CREATE TABLE IF NOT EXISTS search_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    image_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    score REAL,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(query, image_id, model_name, model_revision, embedding_dim),
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    CHECK(label IN ('relevant', 'irrelevant', 'ignored'))
);

CREATE INDEX IF NOT EXISTS idx_search_feedback_query_label
ON search_feedback(query, label);

CREATE INDEX IF NOT EXISTS idx_search_feedback_image_id
ON search_feedback(image_id);

CREATE TABLE IF NOT EXISTS saved_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_saved_views_updated_at
ON saved_views(updated_at);

CREATE TABLE IF NOT EXISTS temporary_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL DEFAULT '',
    color_hex TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_temporary_projects_updated_at
ON temporary_projects(updated_at);

CREATE TABLE IF NOT EXISTS temporary_project_images (
    project_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    intent_label TEXT,
    intent_query TEXT,
    PRIMARY KEY(project_id, image_id),
    FOREIGN KEY(project_id) REFERENCES temporary_projects(id) ON DELETE CASCADE,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_temporary_project_images_project_order
ON temporary_project_images(project_id, sort_order);

CREATE TABLE IF NOT EXISTS inspiration_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    brief TEXT NOT NULL,
    answers TEXT,
    questions_json TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inspiration_projects_updated_at
ON inspiration_projects(updated_at);

CREATE TABLE IF NOT EXISTS inspiration_terms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    query TEXT NOT NULL,
    axis TEXT NOT NULL,
    reason TEXT,
    selected INTEGER NOT NULL DEFAULT 0,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES inspiration_projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_inspiration_terms_project_order
ON inspiration_terms(project_id, sort_order);

CREATE TABLE IF NOT EXISTS creative_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    brief TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'zh',
    provider_name TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    is_pinned INTEGER NOT NULL DEFAULT 0,
    copy_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_creative_projects_updated_at
ON creative_projects(updated_at);

CREATE TABLE IF NOT EXISTS creative_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    parent_id INTEGER,
    title TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    search_query TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES creative_projects(id) ON DELETE CASCADE,
    FOREIGN KEY(parent_id) REFERENCES creative_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_creative_nodes_project_parent_order
ON creative_nodes(project_id, parent_id, sort_order, id);

CREATE TABLE IF NOT EXISTS creative_node_images (
    node_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    intent_label TEXT,
    intent_query TEXT,
    PRIMARY KEY(node_id, image_id),
    FOREIGN KEY(node_id) REFERENCES creative_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_creative_node_images_node_order
ON creative_node_images(node_id, sort_order);

CREATE TABLE IF NOT EXISTS creative_board_layouts (
    project_id INTEGER PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES creative_projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS creative_node_board_layouts (
    node_id INTEGER PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(node_id) REFERENCES creative_nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
