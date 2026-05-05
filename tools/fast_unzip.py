import zipfile
import os
import concurrent.futures
import sys

def unzip_chunk(zip_path, members, dest_dir):
    with zipfile.ZipFile(zip_path) as zf:
        for member in members:
            zf.extract(member, dest_dir)
    return len(members)

def fast_unzip(zip_path, dest_dir, num_workers=16):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    print(f"Opening {zip_path}...")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        
    total_files = len(members)
    print(f"Extracting {total_files} files with {num_workers} workers...")
    
    chunk_size = (total_files + num_workers - 1) // num_workers
    chunks = [members[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(unzip_chunk, zip_path, chunk, dest_dir) for chunk in chunks]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            count = future.result()
            completed += count
            print(f"Progress: {completed}/{total_files} files extracted", end='\r')
    print("\nDone!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fast_unzip.py <zip_file> <dest_dir>")
        sys.exit(1)
        
    zip_file = sys.argv[1]
    dest_dir = sys.argv[2]
    fast_unzip(zip_file, dest_dir)
