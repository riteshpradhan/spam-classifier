#!/usr/bin/perl
use Mail::Mbox::MessageParser;

die "\nUsage:\n\n $0 {filename}\n\n" if ( $#ARGV < 0 );

my $folder_reader = new Mail::Mbox::MessageParser( {'file_name' => $ARGV[0], 'file_handle' =>  new FileHandle($ARGV[0])} );
die $folder_reader unless ref $folder_reader;

while( !$folder_reader->end_of_file() ) 
{ 
  my $email = $folder_reader->read_next_email(); 
  print $$email; 
}
