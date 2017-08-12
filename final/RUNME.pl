: # -*- perl -*-
  eval 'exec perl -S  $0 ${1+"$@"}' 
    if 0;  # if running under some shell
#

system("clear");

$script       = "./predict.py";
@reviewCounts = (5000, 10000, 25000, 50000);
$iters        = 5;

print "Running Yelp Review Predictions\n";

foreach $count(@reviewCounts){
	print "For $count reviews\n";
	for( $i = 0; $i < $iters; $i = $i + 1){
		system("$script -r $count");
	}
}

print "Script Complete\n";