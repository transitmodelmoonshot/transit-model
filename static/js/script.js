function openNav() {
  document.getElementById("menu").style.transform = "translateX(0)";
}

function closeNav() {
  document.getElementById("menu").style.transform = "translateX(100%)";
}

//Restaurant Bookings  321299.837683
//Revenue Hours             1.019891
//Employment             -156.806990
//BC Vaccination Rate  143121.204209
//Season 1              46372.928581
//Season 2             -42873.832536
//Season 3              -3499.096045
function getDateOfISOWeek(w, y) {
    var simple = new Date(y, 0, 1 + (w - 1) * 7);
    var dow = simple.getDay();
    var ISOweekStart = simple;
    if (dow <= 4)
        ISOweekStart.setDate(simple.getDate() - simple.getDay() + 1);
    else
        ISOweekStart.setDate(simple.getDate() + 8 - simple.getDay());
    return ISOweekStart;
}

var week = document.getElementById("week");
var vax_rate = document.getElementById("vax_rate");
var bookings = document.getElementById("bookings");
var revenue_hours = document.getElementById("revenue_hours");
var school_season = document.getElementById("school_season");

var estimate = document.getElementById("mean");
var lower_bound = document.getElementById("lower bound");
var upper_bound = document.getElementById("upper bound");
var week_output = document.getElementById("week_value");
var vax_rate_output = document.getElementById("vax_rate_value");
var bookings_output = document.getElementById("bookings_value");
var revenue_hours_output = document.getElementById("revenue_hours_value");
var school_season_output = document.getElementById("school_season_value");

function predictRidership(vax_rate,bookings,revenue_hours,school_season) {
  var school_ridership;
  if(school_season == 1) {
    school_ridership = 18234.06043;
  }
  if(school_season == 2) {
    school_ridership = -31487.201032;
  }
  if(school_season == 3) {
    school_ridership = 13253.140597;
  }

  var x = Math.round(vax_rate*94118.464520/100+bookings*287083.792445/100+revenue_hours*47.051277+school_ridership-68806.95413600578);
  var x_lower = Math.round(x-x*.13);
  var x_upper = Math.round(x+x*.13);

  lower_bound.innerHTML = x_lower;
  upper_bound.innerHTML = x_upper;
  mean.innerHTML = x;
  console.log(x);
  return
}

function scenario(vax_rate_input,bookings_input,revenue_hours_input,school_season_input) {
  vax_rate.value = vax_rate_input;
  bookings.value = bookings_input;
  revenue_hours.value = revenue_hours_input;
  school_season.value = school_season_input;
  update();
  return
}

function update() {
  predictRidership(parseInt(vax_rate.value),parseInt(bookings.value),parseFloat(revenue_hours.value),parseInt(school_season.value));
  vax_rate_output.innerHTML = vax_rate.value + "%";
  bookings_output.innerHTML = bookings.value + "%";
  revenue_hours_output.innerHTML = revenue_hours.value;
  var s = parseInt(school_season.value);

  if(s == 1) {
    school_season_output.innerHTML = "In Session";
  }
  if(s == 2) {
    school_season_output.innerHTML = "Not in Session";
  }
  if(s == 3) {
    school_season_output.innerHTML = "Holiday Break";
  }
}

update();
