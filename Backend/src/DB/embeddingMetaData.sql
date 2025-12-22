CREATE TABLE employee_face_embedding (
    embedding_id UUID PRIMARY KEY,
    employee_id INT NOT NULL REFERENCES employee(employee_id),
    model_name VARCHAR(50) NOT NULL,          
    embedding_dim INT NOT NULL,                -- 512
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);