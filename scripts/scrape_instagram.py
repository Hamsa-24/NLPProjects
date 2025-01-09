import instaloader
import pandas as pd
import os

def scrape_instagram_comments(username, post_shortcode, output_file="data/raw/instagram_comments.csv"):
    """Scrape comments from an Instagram post."""
    loader = instaloader.Instaloader()
    loader.login("your_username", "your_password")  # Replace with your credentials

    post = instaloader.Post.from_shortcode(loader.context, post_shortcode)
    comments = [{"Comment": comment.text, "Username": comment.owner.username} for comment in post.get_comments()]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(comments).to_csv(output_file, index=False)
    print(f"Comments saved to {output_file}")

if __name__ == "__main__":
    scrape_instagram_comments("target_account", "post_shortcode")
