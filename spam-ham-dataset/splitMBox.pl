#!/usr/bin/perl
use Mail::Mbox::MessageParser;

die "\nUsage:\n\n $0 {filename}\n\n" if ( $#ARGV < 0 );

my $folder_reader = new Mail::Mbox::MessageParser( {'file_name' => $ARGV[0], 'file_handle' =>  new FileHandle($ARGV[0])} );
die $folder_reader unless ref $folder_reader;

$msgCount = 0;
while( !$folder_reader->end_of_file() ) 
{ 
  $msgCount++;
  open(OOO,">split/HAM_${msgCount}.txt");
  my $email = $folder_reader->read_next_email(); 
  print OOO $$email;
  close(OOO); 
}
