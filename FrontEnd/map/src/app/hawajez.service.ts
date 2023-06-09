import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
@Injectable({
  providedIn: 'root'
})
export class HawajezService {

  constructor(private http:HttpClient) { }

  getHawajezPoints(){
    return this.http.get('http://localhost:8000/hawajez_points');
  }
  getHawajezStatus(){
    return this.http.get('http://localhost:8000/hawajez_status');
  }
}
