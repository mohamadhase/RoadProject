// @ts-nocheck
import { Component, OnInit } from '@angular/core';
import * as L from 'leaflet';
import { LatLngTuple } from 'leaflet';
import { poly } from './data.js';
import { poly2 } from './data2.js';
import { poly3 } from './data3.js';
import { Path } from 'leaflet';
import * as turf from '@turf/turf';
import { HawajezService } from './hawajez.service';

// import poly from westbank.js;
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'map';
  map = null;
  palestinePolygon = null;
  palestinePolygon2 = null;
  // define hawajezPoints as array of objects with key string and array of numbers
  hawajezPoints: { [key: string]: [number, number] } = {};
  hawajezMarkers = {};
  hawajezStatus : { [key: string]: string } = {};
  hawajezTime : { [key: string]: string } = {}; // edit it to be date not string
  counter:number = 300;
  current_time = new Date().toLocaleTimeString();
  constructor(private hawajezService:HawajezService) { }
  ngOnInit(): void {
    this.startCounter();
    this.initializeMap();
    this.drawPolygon();
    this.add_status_button();
    this.hawajezService.getHawajezPoints().subscribe((data)=>{
      this.hawajezPoints = data['data'];
      // remove any item in hawajezPoints has empty array value
      for (var key in this.hawajezPoints) {
        if (this.hawajezPoints[key].length == 0) {
          delete this.hawajezPoints[key];
        }
      }
      this.drawHawajezPoints();
      
    });
    this.hawajezService.getHawajezStatus().subscribe((data)=>{
      this.hawajezStatus = data['data'];
      // store the date for each hawajez in hawajezTime
      for (var key in this.hawajezStatus) {
        this.hawajezTime[this.hawajezStatus[key]['location']] = this.hawajezStatus[key]['date'];
      }
      this.hawajezStatus = this.hawajezStatus.reduce((acc, { location, status,date }) => {
        acc[location] = status;
        return acc;
      }, {});
      this.updageHawajezStatus();
      this.updateMarkSign();

    });


  }
  private initializeMap() {
    this.map = L.map('my-map').setView([31.95, 35.23], 8);
    var myAPIKey = 'de36f5ea0ea344919c1e47dec1441f45';
    var isRetina = L.Browser.retina;
    var baseUrl =
      'https://maps.geoapify.com/v1/tile/carto/{z}/{x}/{y}.png?&apiKey=de36f5ea0ea344919c1e47dec1441f45';
    L.tileLayer(baseUrl, {
      maxZoom: 20,
      minZoom: 8,
    }).addTo(this.map);
    

  }

  private drawPolygon() {
    var cord = poly['coordinates'][0];
    var cord2 = poly2['coordinates'][0];
    var cord2 = poly2['coordinates'][0];
  
    this.palestinePolygon = L.geoJSON(poly, {
      coordsToLatLng: function (coords) {
        return new L.LatLng(coords[1], coords[0]);
      },
      style: function (feature) {
        return {
          fillColor: 'red',
          color: 'green', // set the border color to blue
          weight: 2, // set the width of the border
          opacity: 0, // set the opacity of the border
          fillOpacity: 0 // set the opacity of the fill to 0 to make it transparent
        };
      }
    }).addTo(this.map);

    this.palestinePolygon2 = L.geoJSON(poly2, {
      coordsToLatLng: function (coords) {
        return new L.LatLng(coords[1], coords[0]);
      },
      style: function (feature) {
        return {
          fillColor: 'red',
          color: 'green', // set the border color to blue
          weight: 2, // set the width of the border
          opacity: 1, // set the opacity of the border
          fillOpacity: 0 // set the opacity of the fill to 0 to make it transparent
        };
      }
    }).addTo(this.map);

  
    // create a polygon that covers the entire world
    const worldCorners = [
      [-90, -180],
      [-90, 180],
      [90, 180],
      [90, -180],
      [-90, -180],
    ];

    // subtract the Palestine polygon from the world polygon
    let outsidePalestine = turf.difference(turf.polygon([worldCorners]), poly);
    outsidePalestine = turf.difference(outsidePalestine, poly3);

    let outsidePalestine2 = turf.difference(turf.polygon([worldCorners]), poly2);
  
    // create a GeoJSON layer for the area outside Palestine with the desired style
    const outsidePalestineLayer = L.geoJSON(outsidePalestine, {
      style: function (feature) {
        return {
          fillColor: 'black',
          fillOpacity: 0.3,
          color: 'red',
          weight: 1
        };
      }
    }).addTo(this.map);

    const outsidePalestineLayer2 = L.geoJSON(outsidePalestine2, {
      style: function (feature) {
        return {
          fillColor: 'white',
          fillOpacity: 0,
          color: 'green',
          weight: 1
        };
      }
    }).addTo(this.map);
  
    
    this.map.fitBounds(this.palestinePolygon.getBounds());
    this.map.setMaxBounds(this.palestinePolygon.getBounds()*1.5);
    
    this.map.on('moveend', () => {
      if (!this.map.getBounds().intersects(this.palestinePolygon.getBounds())) {
        this.map.panInsideBounds(this.palestinePolygon.getBounds(), { animate: true });
      }
    });
  }
  
  private drawHawajezPoints(){
    // iterate throw each key in the object create a marker and add it to the map and to the hawajezMarkers array
    for (var key in this.hawajezPoints) {
      // create a marker
      var hawajez_point:LatLngTuple = [this.hawajezPoints[key][0], this.hawajezPoints[key][1]];
      const markerIcon = L.icon({
        iconUrl: 'assets/icon.png', // replace with the path to your icon file
        iconSize: [32, 32], // set the size of the icon
        iconAnchor: [16, 32] // set the anchor point of the icon
      });
      // add the marker to the map
      const marker = L.marker(hawajez_point, { icon: markerIcon }).addTo(this.map);
      // add label to the marker contains the key of the object
      const label = key; // replace with your label text
      const popupOptions = { className: 'marker-label' }; // add a class to style the popup label
      marker.bindPopup(label, popupOptions);
      // add the marker to the hawajezMarkers object as a key value pair the key is the key of the object and the value is the marker
      this.hawajezMarkers[key] = marker;
    }
  }
  private updageHawajezStatus(){
    // iterate throw each marker update the popup label to hold the name and the status from the hawajezStatus object
    for (var key in this.hawajezMarkers) {
     var  status = this.hawajezStatus[key];

      if (status ==undefined) {
        status = "لا يوجد معلومات";
      }
      // status-class
      let statusClass = "gray";
      if (status == "مفتوح") {
        console.log (status);
        statusClass = "green";
      } else if (status == "مسكر") {
        statusClass = "red";
      } else if (status == "ازمة") {
        statusClass = "yellow";
      }
       // Construct the popup label with improved styling
       let label = `
       <div class="popup-container" style=text-align:center>
         <h3 class="popup-title">${key}</h3>
         <p class="popup-status ${statusClass}">${status}</p>

     `;
      // Get the current time
      const currentTime = new Date().toLocaleTimeString();

      // Check if this.hawajezTime[key] is defined
      let time: string;
      if (this.hawajezTime[key]) {
        time = this.hawajezTime[key];
      } else {
        time = currentTime;
      }
      label +=  `<div class="popup-time">اخر تحديث: ${time}</div> </div>`
      const popupOptions = { className: 'marker-label' }; // add a class to style the popup label
      this.hawajezMarkers[key].bindPopup(label, popupOptions);
    }
  }

  startCounter() {
    const interval = setInterval(() => {
      this.counter--;
  
      if (this.counter === 0) {
        this.counter = 300;
        console.log("update");
        this.updateData();

      }
    }, 1000);
  }
  updateData(){
    this.hawajezService.getHawajezStatus().subscribe((data)=>{
      this.hawajezStatus = data['data'];
       this.hawajezStatus = this.hawajezStatus.reduce((acc, { location, status }) => {
        acc[location] = status;
        return acc;
      }, {});
      this.updageHawajezStatus();
    });
  }

  getCurrentTime(){
    return new Date().toLocaleTimeString();
  }

  getTimeValue(hajezKey: string): string {
    if (this.hawajezTime[hajezKey]) {
      return this.hawajezTime[hajezKey];
    } else {
      return this.getCurrentTime();
    }
  }
  updateMarkSign(){
    for (var key in this.hawajezMarkers) {
      // print the marker image url

      // print the key
      let status = this.hawajezStatus[key];
      // change the marker size to 32*32
      this.hawajezMarkers[key]._icon.style.width = "20px";
      this.hawajezMarkers[key]._icon.style.height = "20px";

      // if stauts undefined set the marker src to gray.png
      if (status == undefined) {
        this.hawajezMarkers[key]._icon.src = "assets/gray.png";
      }
      else if (status == "مسكر") {
        this.hawajezMarkers[key]._icon.src = "assets/red.png";
      }
      else if (status == "مفتوح") {
        this.hawajezMarkers[key]._icon.src = "assets/green.png";
      }
      else if (status == "ازمة") {
        this.hawajezMarkers[key]._icon.src = "assets/yellow.png";
      }
  }

}
showStatus(){
  // show the div with id hawajez-status if its hidden and hide it if its shown
  if (this.hawajezStatusDiv.nativeElement.style.display == "none") {
    this.hawajezStatusDiv.nativeElement.style.display = "block";
  }
  else {
    this.hawajezStatusDiv.nativeElement.style.display = "none";
  }
}
add_status_button(){
    // add button to the map in the top right corner // this.map is the map object

  // Create a custom control for the status button
  const statusButtonControl = L.control({ position: 'topright' });

  // Define the content and behavior of the control
  statusButtonControl.onAdd = function (map) {
    // Create the button element
    const button = L.DomUtil.create('button', 'status-button');
    button.innerHTML = 'Show Status';

    // Handle the button click event
   // handle the button click event add event listener to the button
    button.addEventListener('click', function () {
      // show the div with id hawajez-status if its hidden and hide it if its shown
      let hawajezStatusDiv = document.getElementById('hawajez-status');

      if (hawajezStatusDiv.style.display == "none") {
        hawajezStatusDiv.style.display = "block"; 
        this.innerHTML = 'Hide Status';
      }
      else {
        hawajezStatusDiv.style.display = "none";
        this.innerHTML = 'Show Status';
      }
    });

    return button;
  };

  // Add the control to the map
  statusButtonControl.addTo(this.map);
}
gotToHajezLocation(hajezKey: string) {
  // get the hajez location from the hawajezLocations object
  console.log(this.hawajezPoints);
  const hajezLocation = this.hawajezPoints[hajezKey];
  console.log(hajezLocation);
  // fly to the hajez location
  this.map.flyTo(hajezLocation, 13); 
  // open the hajez popup
  this.hawajezMarkers[hajezKey].openPopup();
}
getColorClass(value){
  if (value == "مسكر") {
    return "red";
  }
  else if (value == "مفتوح") {
    return "green";
  }
  else if (value == "ازمة") {
    return "yellow";
  }
  return "gray";
}
}
