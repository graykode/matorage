-- SET character_set_client = utf8mb4 ;
--DROP TABLE IF EXISTS compressor CASCADE;
--DROP TABLE IF EXISTS bucket CASCADE;
--DROP TABLE IF EXISTS item_attribute CASCADE;
--DROP TABLE IF EXISTS indexer CASCADE;

CREATE TABLE compressor (
  id integer primary key,
  compressor_level integer not null,
  compressor_lib varchar(255) not null
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

CREATE TABLE item_attribute (
  id integer primary key not null,
  item_size integer not null,
  attribute_shape varchar(255),
  attribute_type varchar(255),
  bucket_id varchar(255), 
  constraint bucket_id foreign key (bucket_id) references bucket(id)
);

CREATE TABLE indexer (
  id integer primary key not null,
  indexer_start bigint not null,
  indexer_length integer not null,
  indexer_name varchar(255) not null,
  bucket_id varchar(255),
  constraint bucket_id foreign key (bucket_id) references bucket(id)
);