-- SET character_set_client = utf8mb4 ;
--DROP TABLE IF EXISTS bucket CASCADE;
--DROP TABLE IF EXISTS attributes CASCADE;
--DROP TABLE IF EXISTS indexer CASCADE;


CREATE TABLE bucket (
  id varchar(255) primary key not null,
  additional text not null,
  dataset_name varchar(255) not null,
  endpoint varchar(255) not null,
  compressor varchar(255) not null,
  sagemaker boolean not null default false
);

CREATE TABLE files(
  id serial primary key not null,
  name varchar(255) not null,
  bucket_id varchar(255),
  constraint bucket_id foreign key (bucket_id) references bucket(id)
);

CREATE TABLE attributes (
  id serial primary key not null,
  name varchar(255) not null,
  type varchar(255) not null,
  shape varchar(255) not null,
  itemsize integer not null,
  bucket_id varchar(255),
  constraint bucket_id foreign key (bucket_id) references bucket(id)
);

CREATE TABLE indexer (
  id serial primary key not null,
  indexer_end bigint not null,
  length integer not null,
  name varchar(255) not null,
  bucket_id varchar(255),
  constraint bucket_id foreign key (bucket_id) references bucket(id)
);