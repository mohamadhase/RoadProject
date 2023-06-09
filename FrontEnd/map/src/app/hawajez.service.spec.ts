import { TestBed } from '@angular/core/testing';

import { HawajezService } from './hawajez.service';

describe('HawajezService', () => {
  let service: HawajezService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(HawajezService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
