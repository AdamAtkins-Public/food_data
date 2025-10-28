CREATE DATABASE grocery_receipt_data;

CREATE TABLE stores (
	store_id SERIAL UNIQUE,
	name VARCHAR(255) NOT NULL,
	location VARCHAR(255) NOT NULL,
	PRIMARY KEY (name, location)
);

CREATE TABLE receipts (
	receipt_id SERIAL UNIQUE,
	store_id INTEGER NOT NULL,
	date DATE NOT NULL,
	FOREIGN KEY (store_id) REFERENCES stores(store_id),
	PRIMARY KEY (store_id, date)
);

CREATE TABLE transaction (
	receipt_id INTEGER NOT NULL,
	general_name VARCHAR(255) NOT NULL,
	abbrv_name VARCHAR(255) NOT	NULL,
	price DECIMAL(5,2) NOT NULL,
	sale BOOLEAN NOT NULL,
	savings DECIMAL(5,2),
	quantity INTEGER NOT NULL,
	unit VARCHAR(5) NOT NULL,
	FOREIGN KEY (receipt_id) REFERENCES receipts(receipt_id)
);
