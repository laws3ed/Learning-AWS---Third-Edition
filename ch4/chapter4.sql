create database laws3eddb;

show databases;

use laws3eddb;

CREATE TABLE Employee (employeeID INT AUTO_INCREMENT, name VARCHAR(255), city VARCHAR(255), PRIMARY KEY(employeeID));

INSERT INTO 
	Employee(name, city)
VALUES
	('AS','SFO'),
	('BS','NYC');

INSERT INTO 
	Employee(name, city)
VALUES
	('CS','LA'),
	('DS','STL');

INSERT INTO 
	Employee(name, city)
VALUES
	('ES','DFW'),
	('FS','AUS');
