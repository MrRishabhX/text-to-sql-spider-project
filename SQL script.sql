-- Create courses table
CREATE TABLE courses (
    course_id INT PRIMARY KEY AUTO_INCREMENT,
    course_name VARCHAR(100) NOT NULL,
    department VARCHAR(50),
    credits INT,
    instructor VARCHAR(100)
);

-- Insert sample data
INSERT INTO courses (course_name, department, credits, instructor) VALUES
('Database Systems', 'Computer Science', 3, 'Dr. Anderson'),
('Machine Learning', 'Computer Science', 4, 'Dr. Smith'),
('Calculus I', 'Mathematics', 4, 'Dr. Johnson'),
('Physics I', 'Physics', 3, 'Dr. Brown'),
('Data Structures', 'Computer Science', 3, 'Dr. Wilson'),
('Linear Algebra', 'Mathematics', 3, 'Dr. Garcia'),
('Software Engineering', 'Computer Science', 4, 'Dr. Martinez'),
('Statistics', 'Mathematics', 3, 'Dr. Davis'),
('Thermodynamics', 'Physics', 4, 'Dr. Miller'),
('Web Development', 'Computer Science', 3, 'Dr. Thompson');

-- Create enrollments table
CREATE TABLE enrollments (
    enrollment_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT,
    course_id INT,
    semester VARCHAR(20),
    year INT,
    grade VARCHAR(2),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

-- Insert sample data
INSERT INTO enrollments (student_id, course_id, semester, year, grade) VALUES
(1, 1, 'Fall', 2024, 'A'),
(1, 2, 'Fall', 2024, 'B+'),
(2, 3, 'Fall', 2024, 'A-'),
(3, 1, 'Fall', 2024, 'B'),
(3, 5, 'Spring', 2024, 'A'),
(4, 4, 'Fall', 2024, 'A-'),
(5, 9, 'Fall', 2024, 'B+'),
(6, 1, 'Spring', 2024, 'A'),
(6, 2, 'Spring', 2024, 'A-'),
(7, 3, 'Fall', 2024, 'B'),
(8, 4, 'Fall', 2024, 'A'),
(9, 9, 'Spring', 2024, 'B+'),
(10, 5, 'Fall', 2024, 'A-');

-- Create departments table
CREATE TABLE departments (
    dept_id INT PRIMARY KEY AUTO_INCREMENT,
    dept_name VARCHAR(50) NOT NULL,
    budget DECIMAL(12,2),
    location VARCHAR(100),
    head_of_department VARCHAR(100)
);

-- Insert sample data
INSERT INTO departments (dept_name, budget, location, head_of_department) VALUES
('Computer Science', 500000.00, 'Building A', 'Dr. Anderson'),
('Sales', 300000.00, 'Building B', 'Mr. Roberts'),
('Marketing', 250000.00, 'Building C', 'Ms. Johnson'),
('Human Resources', 200000.00, 'Building D', 'Ms. Davis'),
('Engineering', 600000.00, 'Building E', 'Dr. Wilson'),
('Mathematics', 350000.00, 'Building F', 'Dr. Garcia'),
('Physics', 400000.00, 'Building G', 'Dr. Brown');

-- Create employees table
CREATE TABLE employees (
    emp_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    department_id INT,
    salary DECIMAL(10,2),
    hire_date DATE,
    position VARCHAR(50),
    email VARCHAR(100),
    FOREIGN KEY (department_id) REFERENCES departments(dept_id)
);

-- Insert sample data
INSERT INTO employees (name, department_id, salary, hire_date, position, email) VALUES
('Alice Johnson', 1, 75000.00, '2022-01-15', 'Software Engineer', 'alice@company.com'),
('Bob Smith', 2, 65000.00, '2021-03-10', 'Sales Manager', 'bob@company.com'),
('Carol Brown', 3, 60000.00, '2020-07-20', 'Marketing Specialist', 'carol@company.com'),
('David Wilson', 1, 80000.00, '2019-05-05', 'Senior Developer', 'david@company.com'),
('Eva Davis', 4, 55000.00, '2023-02-28', 'HR Coordinator', 'eva@company.com'),
('Frank Miller', 5, 85000.00, '2020-09-12', 'Project Manager', 'frank@company.com'),
('Grace Lee', 1, 70000.00, '2022-06-18', 'Full Stack Developer', 'grace@company.com'),
('Henry Taylor', 2, 58000.00, '2021-11-30', 'Sales Representative', 'henry@company.com'),
('Iris Chen', 6, 72000.00, '2021-08-14', 'Research Analyst', 'iris@company.com'),
('Jack Anderson', 7, 78000.00, '2020-04-22', 'Lab Technician', 'jack@company.com');

-- Create orders table
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(100),
    product_name VARCHAR(100),
    quantity INT,
    price DECIMAL(10,2),
    order_date DATE,
    status VARCHAR(20)
);

-- Insert sample data
INSERT INTO orders (customer_name, product_name, quantity, price, order_date, status) VALUES
('John Doe', 'Laptop', 1, 999.99, '2024-01-15', 'Delivered'),
('Jane Smith', 'Mouse', 2, 25.50, '2024-01-16', 'Shipped'),
('Mike Johnson', 'Keyboard', 1, 89.99, '2024-01-17', 'Processing'),
('Sarah Davis', 'Monitor', 1, 299.99, '2024-01-18', 'Delivered'),
('Tom Wilson', 'Headphones', 1, 150.00, '2024-01-19', 'Shipped'),
('Lisa Brown', 'Tablet', 1, 499.99, '2024-01-20', 'Processing'),
('Chris Lee', 'Smartphone', 1, 799.99, '2024-01-21', 'Delivered'),
('Amy Taylor', 'Webcam', 1, 75.99, '2024-01-22', 'Shipped'),
('Ryan Garcia', 'Printer', 1, 199.99, '2024-01-23', 'Processing'),
('Emma Martinez', 'Desk Chair', 1, 249.99, '2024-01-24', 'Delivered');

-- Create books table
CREATE TABLE books (
    book_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(200),
    author VARCHAR(100),
    isbn VARCHAR(20),
    publication_year INT,
    genre VARCHAR(50),
    price DECIMAL(8,2)
);

-- Insert sample data
INSERT INTO books (title, author, isbn, publication_year, genre, price) VALUES
('The Great Gatsby', 'F. Scott Fitzgerald', '978-0-7432-7356-5', 1925, 'Fiction', 12.99),
('To Kill a Mockingbird', 'Harper Lee', '978-0-06-112008-4', 1960, 'Fiction', 14.99),
('1984', 'George Orwell', '978-0-452-28423-4', 1949, 'Science Fiction', 13.99),
('Pride and Prejudice', 'Jane Austen', '978-0-14-143951-8', 1813, 'Romance', 11.99),
('The Catcher in the Rye', 'J.D. Salinger', '978-0-316-76948-0', 1951, 'Fiction', 13.50),
('Lord of the Flies', 'William Golding', '978-0-571-05686-2', 1954, 'Fiction', 12.50),
('Animal Farm', 'George Orwell', '978-0-452-28424-1', 1945, 'Political Fiction', 10.99),
('Brave New World', 'Aldous Huxley', '978-0-06-085052-4', 1932, 'Science Fiction', 14.50);

-- Check all tables
SHOW TABLES;

-- Check row counts
SELECT 'students' as table_name, COUNT(*) as row_count FROM students
UNION ALL
SELECT 'courses', COUNT(*) FROM courses
UNION ALL
SELECT 'enrollments', COUNT(*) FROM enrollments
UNION ALL
SELECT 'departments', COUNT(*) FROM departments
UNION ALL
SELECT 'employees', COUNT(*) FROM employees
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'books', COUNT(*) FROM books;

-- Test some relationships  
SELECT s.name, c.course_name, e.grade 
FROM students s 
JOIN enrollments e ON s.student_id = e.student_id 
JOIN courses c ON e.course_id = c.course_id 
LIMIT 5;

