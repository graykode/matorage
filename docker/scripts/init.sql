-- SET character_set_client = utf8mb4 ;
--DROP TABLE IF EXISTS compressor CASCADE;
--DROP TABLE IF EXISTS bucket CASCADE;
--DROP TABLE IF EXISTS attributes CASCADE;
--DROP TABLE IF EXISTS indexer CASCADE;

CREATE TABLE compressor (
  id integer primary key not null,
  complevel integer not null,
  complib varchar(255) not null
);

CREATE TABLE bucket (
  id varchar(255) primary key not null,
  additional text,
  dataset_name varchar(255) not null,
  endpoint varchar(255) not null,
  filetype text,
  compressor_id integer,
  constraint compressor_id foreign key(compressor_id) references compressor(id)
);

CREATE TABLE attributes (
  id integer primary key not null,
  name varchar(255) not null,
  type varchar(255) not null,
  shape varchar(255) not null,
  itemsize integer not null,
  bucket_id varchar(255),
  constraint bucket_id foreign key (bucket_id) references bucket(id)
);

CREATE TABLE indexer (
  id integer primary key not null,
  start bigint not null,
  length integer not null,
  name varchar(255) not null,
  bucket_id varchar(255),
  constraint bucket_id foreign key (bucket_id) references bucket(id)
);